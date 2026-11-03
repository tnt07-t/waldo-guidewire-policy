"""Planned CNN-GRU policy over the six-frame visual history; not implemented yet.

Wraps the per-image CNN from cnn.py (ResNet-18 with its classification head
removed, so each image becomes one 512-value feature vector) in a GRU that
reads the chronological history f1..f6 and predicts ADVANCE or RETRACT.
"""

# Inputs per case (batch size B, after model-input preparation):
#   goal      (B, 3, H, W)     target-marked reference; NOT a history step
#   history   (B, 6, 3, H, W)  f1..f6 in chronological order, f6 = now
#   depth     (B, 6)           depth_relative_mm, last value is 0
#   offsets   (B, 6)           history_offsets_s, last value is 0; spacing varies
# Output: logits (B, 2) or (B, 1), one decision for the current moment.

# Step 1 - Encode every history frame with the same CNN.
# A CNN only takes 4-D input, so fold the time axis into the batch axis, run the
# CNN once, then unfold. Every frame shares the same weights.
#
#   B, T, C, H, W = history.shape                       # T = 6
#   frames = history.reshape(B * T, C, H, W)
#   feats = cnn(frames)                                 # (B * T, 512)
#   feats = feats.reshape(B, T, 512)                    # (B, 6, 512)

# Step 2 - Attach per-step metadata so the GRU sees motion, not just pixels.
# Frames are not evenly spaced, so the GRU cannot assume a fixed time step;
# give it each frame's time offset and relative depth alongside its features.
# Scale both to roughly unit range first (e.g. divide by a train-set std).
#
#   meta = torch.stack([depth, offsets], dim=-1)        # (B, 6, 2)
#   steps = torch.cat([feats, meta], dim=-1)            # (B, 6, 514)
#   steps = proj(steps)                                 # optional Linear(514, 256)
#
# The optional projection shrinks 512 -> 256 before the GRU, which cuts GRU
# parameters substantially; worth it with only 416 cases.

# Step 3 - Run the GRU over time to summarize the visual history.
#
#   gru = nn.GRU(input_size=256, hidden_size=256, num_layers=1, batch_first=True)
#   outputs, h_n = gru(steps, h0)                       # h_n: (1, B, 256)
#   history_summary = h_n[-1]                           # (B, 256), state after f6
#
# Because the sequence runs oldest -> newest, the final hidden state is
# "everything seen so far, as of now", which is exactly what the decision needs.
# Every case has exactly six frames, so no padding/pack_padded_sequence needed.
# Keep it unidirectional: a backward pass adds parameters without adding
# information the final forward state lacks for a "now" decision.

# Step 4 - Bring in the goal reference (encode it separately, do not append it
# as a 7th step, since it is not part of the timeline).
# Reuse the same CNN to encode it: goal_feat = cnn(goal) -> (B, 512).
# Two reasonable ways to fuse it, pick one:
#   a) Late fusion (simplest, start here):
#        fused = torch.cat([history_summary, goal_feat], dim=-1)   # (B, 768)
#   b) Goal-conditioned GRU: initialize the hidden state from the goal so the
#      GRU reads the history "knowing where it is going":
#        h0 = torch.tanh(goal_to_h0(goal_feat)).unsqueeze(0)       # (1, B, 256)
#      then pass h0 into the GRU in Step 3 and use history_summary directly.

# Step 5 - Classification head.
#
#   head = nn.Sequential(nn.Dropout(0.3), nn.Linear(fused_dim, 128), nn.ReLU(),
#                        nn.Dropout(0.3), nn.Linear(128, 2))
#   logits = head(fused)
#
# Labels are imbalanced (0: 157, 1: 259); use class weights in
# CrossEntropyLoss (or pos_weight with a single-logit BCEWithLogitsLoss).

# Training notes.
# - Freeze the CNN at first. 416 cases is far too few to fine-tune ResNet-18
#   end to end without overfitting. With a frozen CNN, put it in eval() mode
#   (BatchNorm statistics must not update) and wrap it in torch.no_grad().
# - With a frozen CNN, precompute and cache the (6, 512) history features and
#   the (512,) goal feature per case once; GRU training then runs in seconds
#   per epoch and only the projection, GRU, and head are trained.
# - Once that works, optionally unfreeze ResNet-18 layer4 with a much lower
#   learning rate (e.g. 1e-5 vs 1e-3 for the GRU/head) and compare on validation.
# - Baseline to beat: the same head on cnn(f6) + goal_feat alone (no GRU). If the
#   GRU does not beat it, the history is not being used.

# TODO: agree on input resolution and normalization shared with cnn.py.
# TODO: define CNNGRUPolicy(torch.nn.Module) with forward(goal, history, depth, offsets).
# TODO: pick goal fusion (a) or (b) and the depth/time scaling.
