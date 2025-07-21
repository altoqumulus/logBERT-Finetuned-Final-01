#* Why DIST token at pos 0? It aggregates the entire window and feeds VHM.
#* Why mask_ratio = 0.3? Paper sweeps 0.1–0.5 and finds 0.3 a sweet spot.
#* Why MASK_IDX = ‑100? PyTorch’s nn.CrossEntropyLoss ignores ‑100.

import torch, pickle, random

'''
MASK_IDX = -100          # value we’ll use to identify masked positions
DIST_IDX = 0             # h_DIST token at position 0

def random_mask(seq, mask_ratio=0.3):
    """Return (masked_seq, labels) as tensors."""
    seq = seq.clone()
    labels = torch.full_like(seq, fill_value=-100)   # BCE ignores -100
    # We never mask DIST (pos 0)
    num_can_mask = seq.size(0) - 1
    k = max(1, int(num_can_mask * mask_ratio))
    mask_positions = random.sample(range(1, seq.size(0)), k=k)
    for pos in mask_positions:
        labels[pos] = seq[pos]          # remember original token
        seq[pos] = MASK_IDX             # replace with MASK
    return seq, labels
'''

VOCAB_SIZE  = 49        # size of the vocabulary, including DIST and MASK
IGNORE_IDX   = -100           # used ONLY for the loss
MASK_TOKEN   = VOCAB_SIZE - 1 # a real token, gets an embedding

def random_mask(seq, mask_ratio=0.3):
    seq = seq.clone()
    labels = torch.full_like(seq, IGNORE_IDX)
    k = max(1, int((seq.size(0)-1) * mask_ratio))
    positions = random.sample(range(1, seq.size(0)), k)
    for pos in positions:
        labels[pos] = seq[pos]
        seq[pos]    = MASK_TOKEN
    return seq, labels

    
class WindowDataset(torch.utils.data.Dataset):
    """Loads pickled windows; if labels exist returns (seq,label)."""
    def __init__(self, path, with_labels=False, mask_ratio=0.3):
        self.data = pickle.load(open(path, "rb"))
        self.with_labels = with_labels
        self.mask_ratio = mask_ratio

    def __len__(self):  return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        if self.with_labels:
            seq, label = item
            seq = torch.tensor(seq, dtype=torch.long)
            return seq, torch.tensor(label, dtype=torch.long)
        else:
            seq = torch.tensor(item, dtype=torch.long)
            return seq

def collate_mask_fn(batch, mask_ratio=0.3):
    """Batchs variable structure from Dataset & applies masking."""
    if isinstance(batch[0], tuple):          # test set
        seqs, labels = zip(*batch)
        seqs = torch.stack(seqs)
        return seqs, torch.tensor(labels)
    else:                                    # train / val
        seqs = torch.stack(batch)
        masked, tgt = [], []
        for seq in seqs:
            mseq, lbl = random_mask(seq, mask_ratio)
            masked.append(mseq)
            tgt.append(lbl)
        return torch.stack(masked), torch.stack(tgt)
