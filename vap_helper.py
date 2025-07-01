import torch

def get_va_states_by_speaker_bin_mask(vap_wrapper, speaker_idx=0, bin_mask=None):
    """
    Get the probability of a specific speaker having voice activity in specified bins.
    
    Args:
        speaker_idx: 0 for speaker A, 1 for speaker B 
        bin_mask: List of 4 booleans indicating which bins to *CONSIDER* [bin0, bin1, bin2, bin3] (to consider != to have voice activity)
                  If None, considers all bins [True, True, True, True]
    
    Returns:
        Probability tensor of shape [batch, frames]
    """
    if bin_mask is None:
        bin_mask = [True, True, True, True]  # All 4 bins
    
    # Get the codebook from VAP wrapper's objective
    codebook = vap_wrapper.model.objective.codebook
    
    # Generate all possible states where the specified speaker is active in the specified bins
    # and sum their probabilities
    
    # Create all possible 4-bin patterns for the target speaker
    target_speaker_patterns = []
    other_speaker_patterns = []
    
    # Generate all 16 possible bin patterns for EACH speaker (2^4 = 16)
    for i in range(16):  # 2^4 combinations

        ## For i in range(16):  # i goes from 0 to 15, i.e., 4 bits maximum
        ## By shifting 0, 1, 2, 3 bits (>> j), we push the j-th bit in i to the least significant position (rightward)
        ## Then the & operation with 1 extracts that bit, resulting in a binary pattern
        ## So putting together, i >> j & 1 basically checks if the j-th bit in i is set (1) or not (0)
        pattern = [(i >> j) & 1 for j in range(4)]  # Convert to binary pattern
        
        ## Check if this pattern matches our bin_mask constraint for target speaker
        ## Since we will marginalize out irrelevant bins, we only need the pattern to match with bin_mask at the positions where bin_mask is True
        matches_constraint = all(
            pattern[j] == 1 if bin_mask[j] else True 
            for j in range(4)
        )
        
        # For the target speaker, we only want patterns that match the bin_mask constraint
        if matches_constraint:
            target_speaker_patterns.append(pattern)
        
        # For the other speaker, we want all possible patterns (to be marginalized out)
        other_speaker_patterns.append(pattern)
    
    # Generate all valid state combinations that we will sum over later for marginalization
    valid_states = []
    for target_pattern in target_speaker_patterns:
        for other_pattern in other_speaker_patterns:
            # Create the full state pattern (2 speakers x 4 bins) by concatenating the two patterns
            # Based on VAP codebase: Speaker A (idx=0) comes first, then Speaker B (idx=1)
            if speaker_idx == 0:  # Speaker A is target
                full_pattern = torch.tensor([target_pattern + other_pattern], dtype=torch.float32)
            else:  # Speaker B is target  
                full_pattern = torch.tensor([other_pattern + target_pattern], dtype=torch.float32)
            
            # Convert to state index via the code book
            flattened_pattern = full_pattern.view(1, 2, 4)
            # Move to the correct device if needed
            flattened_pattern = flattened_pattern.to(vap_wrapper.device)
            state_idx = codebook.encode(flattened_pattern)
            valid_states.append(state_idx.item())
    
    # Valid states to sum over
    valid_states = torch.tensor(valid_states, dtype=torch.long)

    ## Example: you can marginalize by the following line
    # marginal_prob = vap_result_probs[..., valid_states].sum(dim=-1)
    
    return valid_states
