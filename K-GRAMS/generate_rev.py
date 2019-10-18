import torch
from config import config, device
from torch.utils.data import DataLoader
from KGRAMSData import KGRAMSData, KGRAMSEvalData, KGRAMSTrainData

def get_reviews_from_batch(batch_rev, idx2word_dict):
    review_list = []
    for rev in batch_rev:
        review = []
        for idx in rev:
            review.append(idx2word_dict[idx])
        rev_str = " ".join(review)
        review_list.append(rev_str)
    return review_list



def generate_reviews(config, model, data_loader, idx2word_dict):
    all_actual_reviews = []
    all_generated_reviews = []
    for batch in data_loader:
        target_user_id = batch[0].to(device)
        target_item_id = batch[1].to(device)
        user_reviews = batch[4].to(device)
        item_reviews = batch[5].to(device)
        target_reviews = torch.stack(batch[3], dim=1).view(config.batch_size, -1).to(device)
        review_user_ids = torch.stack(batch[6], dim=1).view(config.batch_size, -1).to(device)
        review_item_ids = torch.stack(batch[7], dim=1).view(config.batch_size, -1).to(device)
        rating_pred, word_idx_seq = model(user_ids=target_user_id,
                                          user_reviews=user_reviews,
                                          user_ids_of_reviews=review_user_ids,
                                          item_ids=target_item_id,
                                          item_reviews=item_reviews,
                                          item_ids_of_reviews=review_item_ids,
                                          target_reviews_x=None,
                                          mode="test")


        actual_review = get_reviews_from_batch(target_reviews.tolist(), idx2word_dict)
        generated_review = get_reviews_from_batch(word_idx_seq, idx2word_dict)
        all_actual_reviews.extend(actual_review)
        all_generated_reviews.extend(generated_review)
    return all_actual_reviews, all_generated_reviews


if __name__ == "__main__":
    saved_model_path = "models/model_1000.pt"
    data_path = config.root_dir + config.data_set_name
    dataset_test = KGRAMSEvalData(data_path, config.review_length, mode="test")
    data_loader = DataLoader(dataset_test, config.batch_size, shuffle=True, num_workers=0, drop_last=True, timeout=0)
    model = torch.load(saved_model_path)
    orig_reviews, generated_reviews = generate_reviews(config, model, data_loader, dataset_test.idx2word)
    orig_file = open("orig.txt", "w")
    orig_file.write("\n".join(orig_reviews))
    gen_file = open("generated.txt", "w")
    gen_file.write("\n".join(generated_reviews))
