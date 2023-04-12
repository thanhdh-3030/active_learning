import torch
from src.trainer.config import *

@torch.no_grad()
def core_set_selection_decoder(emb_model,labeled_flags,train_dataset,budget,device='cuda'):
    # dataset.initialize_labels(num=100)
    emb_model.train()
    print(device)
    emb_model.to(device)
    pool_loader=torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=1)
    # get embeddings
    embeddings=[]
    old_labeled_flags=labeled_flags.copy()
    for index,batch in enumerate(tqdm(pool_loader)):
        images=batch['image'].to(device)
        # embedding=emb_model.forward(images)
        embedding=emb_model.forward_decoder(images)
        # embeddings.append(embedding.squeeze(0).squeeze(0).cpu().flatten().numpy())
        embeddings.append(embedding.squeeze(0).cpu().flatten().numpy())
    embeddings=np.array(embeddings).astype(np.float32)
    dist_mat = np.matmul(embeddings, embeddings.transpose())
    sq = np.array(dist_mat.diagonal()).reshape(len(labeled_flags), 1)
    dist_mat *= -2
    dist_mat += sq
    dist_mat += sq.transpose()
    dist_mat = np.sqrt(dist_mat)

    mat = dist_mat[~labeled_flags, :][:,labeled_flags]

    for i in tqdm(range(budget), ncols=100):
        mat_min = mat.min(axis=1)
        q_idx_ = mat_min.argmax()
        q_idx = np.arange(len(train_dataset))[~labeled_flags][q_idx_]
        labeled_flags[q_idx] = True
        mat = np.delete(mat, q_idx_, 0)
        mat = np.append(mat, dist_mat[~labeled_flags, q_idx][:, None], axis=1)
    query_idxs=np.arange(len(train_dataset))[(old_labeled_flags^labeled_flags)]
    # query_samples=pool_samples[query_idxs]
    return query_idxs,labeled_flags