    # dim == 512
    # positional embedding for point cloud -> text
    self.pc_text_pos_emb = nn.Embedding(pc_seq_len + text_seq_len + 1, dim) 
    # positional embedding for point cloud -> shape program
    self.pc_prog_pos_emb = nn.Embedding(pc_seq_len + prog_seq_len + 1, dim)