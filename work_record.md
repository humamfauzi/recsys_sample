## ISSUE-001: Explosive Growth in Latent Factor [In Progress]
**Module**: train/train.py:ALSModel

**Priority**: High

**Created**: 2025-07-11

**Description**: The latent factor in the ALS model is growing exponentially, leading to singluar matrix and instability.


**Log Entries**:
- [info] The metadata have a large, unnormalized number. We need to find the source.
- [info] Preprocessing does not delete the ID. It carried to latent calculation. Try dropping the ID column for both user and item.
- [info] Decide not to drop the ID column. Instead, we alter the metadata range since it is the one responsible to picking metadata properties.
- [info] Apprently there are other sources of explosion. First and second iteration seems fine, but the third iteration is already exploding. Commit the changes here.
- [info] Current assumption is that there are some users who explodes the latent factor. 
- [info] The user would have large latent factor, in turn it would affect all latent product it used.
- [info] After several iterations, this would cascade to all users and items.
- [info] Possible users are 166, 171, 202 
- [info] There are no significant difference. 
- [info] Collaborative filtering does induce value propagation value between users and items that does not have any interaction.
- [info] However, this would make it harder to pin where things go wrong in iteration.
- [info] It seems our loss function does not count for metadata.
- [info] Something wrong in latent item calculation. The explosive latent factor most of the time comes from item latent factor.
- [info] Based on the log, it seems that item with small number of interactions has large latent factor.
- [info] This is because the sparse matrix introduce a ill conditioned matrix
- [info] Adaptive regularization seems to have a good effect. It able to reduce the latent matrix eigenvalues ratio.
- [info] Laten matrix eigenvalues max min ratio is a good measure of how ill conditioned the matrix is.
- [info] The problem still persists. The latent factor calculation still dominates the rating prediction.
- [info] We can do normalization of latent factor. This ensures that latent factor dot product.
- [info] We do this when updating the latent factor. So it still has the change and would calculate the loss **after** normalized.
- [info] The latent is properly clamped but now the the bias getting wilder

**Action Items**:
- [X] Add metadata in loss calculation
- [X] Implement adaptive regularization
- [ ] Add latent normalization

## ISSUE-002: Range and Column Constant [To Do]