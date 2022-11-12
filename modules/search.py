import torch, operator
import torch.nn.functional as F
from queue import PriorityQueue
from collections import namedtuple



class Beam:
    def __init__(self, config, model, tokenizer):
        super(Beam, self).__init__()
        self.beam_size = 4
        self.model = model
        self.tokenizer = tokenizer
        self.device = config.device
        self.bos_idx = config.bos_idx
        self.eos_idx = config.eos_idx
        self.pad_idx = config.pad_idx
        self.model_name = config.model_name
        
        if self.model.training:
            self.model.eval()
        
        if self.model_name != 'transformer':
            self.Node = namedtuple('Node', ['prev_node', 'hidden', 'log_prob', 'pred', 'preds', 'length'])
        elif self.model_name == 'transformer':
            self.Node = namedtuple('Node', ['prev_node', 'pred', 'log_prob', 'length'])


    def get_score(self, node, max_repeat, min_length=5, alpha=1.2):
        repeat = max([node.pred.tolist().count(token) for token in node.pred.tolist() if token != self.pad_idx])

        if repeat > max_repeat + 5:
            repeat_penalty = -1
        else:
            repeat_penalty = 1
        
        length_penalty = ((node.length + min_length) / (1 + min_length)) ** alpha
        score = node.log_prob / length_penalty
        score = score * repeat_penalty
        return score


    def get_output(self, top_node):
        if self.model_name != 'transformer':
            output = []
            while top_node.prev_node is not None:
                output.append(top_node.pred.item())
                top_node = top_node.prev_node
            return output[::-1]

        elif self.model_name == 'transformer':
            return top_node.pred.tolist()


    def get_nodes(self, hiddens=None):
        Node = self.Node
        nodes = PriorityQueue()
        start_tensor = torch.LongTensor([self.bos_idx]).to(self.device)

        if self.model_name != 'transformer':
            start_node = Node(prev_node = None, 
                              hiddens = hiddens, 
                              log_prob = 0.0, 
                              pred = start_tensor, 
                              preds = [self.bos_idx],
                              length = 0)
        elif self.model_name == 'transformer':
            start_node = Node(prev_node = None,
                              pred = start_tensor,
                              log_prob = 0.0,
                              length = 0)

        for _ in range(self.beam_size):
            nodes.put((0, start_node))        
        return Node, nodes, [], []    


    def get_input_params(self, input_seq):
        input_tokens = self.tokenizer.encode(input_seq)
        input_tensor = torch.LongTensor(input_tokens).unsqueeze(0).to(self.device)
        max_len = len(input_tokens) + 30
        max_repeat = max([input_tokens.count(token) for token in input_tokens if token != self.pad_idx])
        return input_tensor, max_len, max_repeat        




class RNNSearch(Beam):
    def __init__(self, config, model, tokenizer):
        super(RNNSearch, self).__init__()

    def beam_search(self, input_seq, topk=1):
        input_tensor, max_len, max_repeat = self.get_input_params(input_seq)
        Node, nodes, end_nodes, top_nodes = self.get_nodes(hiddens=self.model.encoder(input_tensor))

        for t in range(max_len):
            curr_nodes = [nodes.get() for _ in range(self.beam_size)]
            
            for curr_score, curr_node in curr_nodes:
                if curr_node.pred.item() == self.eos_idx and curr_node.prev_node != None:
                    end_nodes.append((curr_score, curr_node))
                    continue
                    
                out, hidden = self.model.decoder(curr_node.pred, curr_node.hidden)
                logits, preds = torch.topk(out, self.beam_size)
                log_probs = F.log_softmax(logits, dim=-1)

                for k in range(self.beam_size):
                    pred = preds[0][k].view(1)
                    log_prob = log_probs[0][k].item()

                    next_node = Node(prev_node = curr_node,
                                     hidden = hidden,
                                     log_prob = curr_node.log_prob + log_prob,
                                     pred = pred.contiguous(),
                                     preds = curr_node.preds + [pred.item()],
                                     length = curr_node.length + 1)
                    
                    next_score = self.get_score(next_node, max_repeat)
                    nodes.put((next_score, next_node))    

                if not t:
                    break

        if len(end_nodes) == 0:
            _, top_node = nodes.get()
        else:
            _, top_node = sorted(end_nodes, key=operator.itemgetter(0), reverse=True)[0]
        
        return self.get_output(top_node)
    

    def greedy_search(self, input_seq):
        input_tensor, max_len, _ = self.get_input_params(input_seq)
        
        return



class TransSearch(Beam):
    def __init__(self, config, model):
        super(TransSearch, self).__init__()

    def beam_search(self, input_seq, topk=1):
        Node, nodes, end_nodes, top_nodes = self.get_nodes()
        input_tensor, max_len, max_repeat = self.get_input_params(input_seq)

        e_mask = self.model.pad_mask(input_tensor)
        memory = self.model.encoder(input_tensor, e_mask)
        
        for t in range(max_len):
            curr_nodes = [nodes.get() for _ in range(self.beam_size)]

            for curr_score, curr_node in curr_nodes:
                if curr_node.pred[-1].item() == self.eos_idx and curr_node.prev_node != None:
                    end_nodes.append((curr_score, curr_node))
                    continue
                
                d_input = curr_node.pred.unsqueeze(0) 
                d_mask = self.model.dec_mask(d_input)
                dec_out = self.model.decoder(d_input, memory, e_mask)
                out = self.model.fc_out(dec_out)[:, -1, :]
                
                logits, preds = torch.topk(out, self.beam_size)
                logits, preds = logits.squeeze(1), preds.squeeze(1)
                log_probs = F.log_softmax(logits, dim=-1)

                for k in range(self.beam_size):
                    pred = preds[0][k].view(1)
                    log_prob = log_probs[0][k].item()
                    new_pred = torch.cat([curr_node.pred, pred], dim=-1)           
                    
                    next_node = Node(prev_node = curr_node,
                                     pred = new_pred,
                                     log_prob = curr_node.log_prob + log_prob,
                                     length = curr_node.length + 1)
                    next_score = self.get_score(next_node, max_repeat, self.pad_idx)                
                    nodes.put((next_score, next_node))
                
                if not t:
                    break

        if len(end_nodes) == 0:
            _, top_node = nodes.get()
            print(f"Total End Nodes: {len(end_nodes)}")
        else:
            print(f"Total End Nodes: {len(end_nodes)}")
            _, top_node = sorted(end_nodes, key=operator.itemgetter(0), reverse=True)[0]
        
        return self.get_output(top_node)        


    def greedy_search(self, input_seq):
        input_tensor, max_len, _ = self.get_input_params(input_seq)

        e_mask = self.model.pad_mask(input_tensor)
        memory = self.model.encoder(input_tensor, e_mask)

        return