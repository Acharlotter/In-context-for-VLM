import torch
import clip
from PIL import Image
import os
import json
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import faiss
import os
import random

os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("/hy-tmp/models/clip/ViT-B-32.pt", device=device)

def extract_features(filenames):
    features = []

    for img in tqdm(filenames):
        imge_input = preprocess(Image.open(img)).unsqueeze(0).to(device)
        with torch.no_grad():
            features.append(model.encode_image(imge_input))
    features = torch.stack(features).squeeze(1)

    #print(features.shape)
    return features



########################VQAv2数据集实验########################



def retrieve_vqa_vqav2():
    train_data_dir = "/hy-tmp/mscoco2014/train2014"
    val_data_dir = "/hy-tmp/mscoco2014/val2014"

    # prepare train data
    train_captions = json.load(open("/hy-tmp/VQA_VQAv2/retrieval/vqav2/v2_OpenEnded_mscoco_train2014_questions.json", 'r'))["questions"]  #下载数据集，改路径
    train_answers = json.load(open("/hy-tmp/VQA_VQAv2/retrieval/vqav2/v2_mscoco_train2014_annotations.json", 'r'))["annotations"]
    # prepare test data
    test_captions = json.load(open("/hy-tmp/VQA_VQAv2/retrieval/vqav2/v2_OpenEnded_mscoco_val2014_questions.json", 'r'))["questions"]
    test_answers = json.load(open("/hy-tmp/VQA_VQAv2/retrieval/vqav2/v2_mscoco_val2014_annotations.json", 'r'))["annotations"]

    if not os.path.exists('/hy-tmp/VQA_VQAv2/retrieval/retrieval_results/vqa'):
        os.mkdir('/hy-tmp/VQA_VQAv2/retrieval/retrieval_results/vqa') #改路径

    # extract train feature
    train_ids = []
    train_img_ids = []
    train_filenames = []
    for idx, caption in enumerate(train_captions):
        train_ids.append(caption["question_id"])
        train_img_ids.append(caption["image_id"])
        train_name = "COCO_train2014_" + str(caption["image_id"]).zfill(12) + ".jpg"
        train_filenames.append(os.path.join(train_data_dir, train_name))

    if not os.path.exists('/hy-tmp/cp_npy/train_image_features.npy'):
        train_features = extract_features(train_filenames)
        np.save("/hy-tmp/cp_npy/train_image_features.npy", train_features.to("cpu"))

    # text encode
    text_features = []
    if not os.path.exists('/hy-tmp/cp_npy/train_text_features.npy'):
        text_inputs = clip.tokenize([i['question'] for i in train_captions])
        for i in tqdm(text_inputs):
            with torch.no_grad():
                text_features.append(model.encode_text(i.unsqueeze(0).to(device)))
        text_features = torch.stack(text_features).squeeze(1)
        np.save("/hy-tmp/cp_npy/train_text_features.npy", text_features.to("cpu"))
    # qa
    text_features = []
    if not os.path.exists('/hy-tmp/cp_npy/train_text_features_qa.npy'):
        print("Extract train qa...")
        text_inputs = clip.tokenize([i['question'] + j['multiple_choice_answer'] for (i, j) in zip(train_captions, train_answers)])
        for i in tqdm(text_inputs):
            with torch.no_grad():
                text_features.append(model.encode_text(i.unsqueeze(0).to(device)))
        text_features = torch.stack(text_features).squeeze(1)
        np.save("/hy-tmp/cp_npy/train_text_features_qa.npy", text_features.to("cpu"))
        print("Done train qa.")

    # read features
    train_image_features = torch.from_numpy(np.load("/hy-tmp/cp_npy/train_image_features.npy")).to(device)
    train_text_features = torch.from_numpy(np.load("/hy-tmp/cp_npy/train_text_features.npy")).to(device)
    train_text_features_qa = torch.from_numpy(np.load("/hy-tmp/cp_npy/train_text_features_qa.npy")).to(device)
    print(train_image_features.shape, train_text_features.shape, train_text_features_qa.shape)


    # extract test feature
    test_ids = []
    test_img_ids = []
    test_filenames = []
    for caption in test_captions:
        test_ids.append(caption["question_id"])
        test_img_ids.append(caption["image_id"])
        test_name = "COCO_val2014_" + str(caption["image_id"]).zfill(12) + ".jpg"
        test_filenames.append(os.path.join(val_data_dir, test_name))
    if not os.path.exists('/hy-tmp/cp_npy/test_image_features.npy'):
        test_features = extract_features(test_filenames)
        np.save("/hy-tmp/cp_npy/test_image_features.npy", test_features.to("cpu"))

    # text encode
    text_features = []
    if not os.path.exists('/hy-tmp/cp_npy/test_text_features.npy'):
        text_inputs = clip.tokenize([i['question'] for i in test_captions])
        for i in tqdm(text_inputs):
            with torch.no_grad():
                text_features.append(model.encode_text(i.unsqueeze(0).to(device)))
        text_features = torch.stack(text_features).squeeze(1)
        np.save("/hy-tmp/cp_npy/test_text_features.npy", text_features.to("cpu"))

    text_features = []
    if not os.path.exists('/hy-tmp/cp_npy/test_text_features_qa.npy'):
        print("Extract test qa...")
        text_inputs = clip.tokenize([i['question'] + j['multiple_choice_answer'] for (i, j) in zip(test_captions, test_answers)])
        for i in tqdm(text_inputs):
            with torch.no_grad():
                text_features.append(model.encode_text(i.unsqueeze(0).to(device)))
        text_features = torch.stack(text_features).squeeze(1)
        np.save("/hy-tmp/cp_npy/test_text_features_qa.npy", text_features.to("cpu"))
        print("Done test qa.")

    # read features
    test_image_features = torch.from_numpy(np.load("/hy-tmp/cp_npy/test_image_features.npy")).to(device)
    test_text_features = torch.from_numpy(np.load("/hy-tmp/cp_npy/test_text_features.npy")).to(device)
    test_text_features_qa = torch.from_numpy(np.load("/hy-tmp/cp_npy/test_text_features_qa.npy")).to(device)
    print(test_image_features.shape, test_text_features.shape, test_text_features_qa.shape)

    # softmax
    test_image_features /= test_image_features.norm(dim=-1, keepdim=True)
    test_text_features /= test_text_features.norm(dim=-1, keepdim=True)
    test_text_features_qa /= test_text_features_qa.norm(dim=-1, keepdim=True)
    train_image_features /= train_image_features.norm(dim=-1, keepdim=True)
    train_text_features /= train_text_features.norm(dim=-1, keepdim=True)
    train_text_features_qa /= train_text_features_qa.norm(dim=-1, keepdim=True)

    # similarity_imgs = (100.0 * test_image_features @ train_image_features.T).softmax(dim=-1)
    # similarity_caps = (100.0 * test_text_features @ train_text_features.T).softmax(dim=-1)

    def similarity_retrieval():
        import faiss
        output_all = defaultdict(dict)

        def SI_retrieval():
            index = faiss.IndexFlatL2(512)
            index.add(train_image_features.cpu())
            print(index.ntotal)
            # SI
            if not os.path.exists('/hy-tmp/Iteration/image_image_indices_imgs.npy'):
                values_imgs, indices_imgs = index.search(test_image_features.cpu(), 2000)
                np.save("/hy-tmp/Iteration/image_image_indices_imgs.npy", indices_imgs)
                np.save("/hy-tmp/Iteration/image_image_values_imgs.npy", values_imgs)
            else:
                values_imgs = np.load("/hy-tmp/Iteration/image_image_values_imgs.npy")
                indices_imgs = np.load("/hy-tmp/Iteration/image_image_indices_imgs.npy")
            for idx, (value, index) in enumerate(zip(values_imgs, indices_imgs)):
                output_imgs = []
                output_imgs_new = []
                for id, val in zip(index[:32], value[:32]):
                    output_imgs.append([train_ids[id], val.tolist(), id])
                output_all[test_ids[idx]]["SI"] = output_imgs
                # #SI+_1
                # candidate_imgs = defaultdict(list)
                # for id, val in zip(index, value):
                #     img_id = str(train_ids[id])[:6] # str
                #     candidate_imgs[img_id].append([train_ids[id], val.tolist(), id])
                #     if len(candidate_imgs) >= 32:
                #         break
                # train_text_features_new_idx = []
                # for candidate_img in candidate_imgs.values():
                #     train_text_features_new_idx += [feature[2] for feature in candidate_img]
                # train_text_features_new = train_text_features[train_text_features_new_idx]
                # similarity_img_cap = (100.0 * test_text_features[idx] @ train_text_features_new.T).softmax(dim=-1)
                # # values_img_cap, indices_img_cap = similarity_img_cap[0].topk(32)
                # values_img_cap_1, indices_img_cap_1 = similarity_img_cap.topk(len(similarity_img_cap))
                # values_img_cap_, indices_img_cap_ = similarity_img_cap.topk(32)
                # indices_img_cap = index[indices_img_cap_.tolist()]
                # indices_img_cap_1 = index[indices_img_cap_1.tolist()]
                # output_img_cap = []
                # for id, val in zip(indices_img_cap, values_img_cap_):
                #     output_img_cap.append([train_ids[id], val.tolist(), id])
                # output_all[test_ids[idx]]["SI_1"] = output_img_cap
                # #SI+_2
                # output_img_cap = []
                # all_ready_in = []
                # for id, val in zip(indices_img_cap_1, values_img_cap_1):
                #     if str(train_ids[id])[:6] not in all_ready_in:
                #         output_img_cap.append([train_ids[id], val.tolist(), id])
                #         all_ready_in.append(str(train_ids[id])[:6])
                #     if len(all_ready_in) >= 32:
                #         break
                # output_all[test_ids[idx]]["SI_2"] = output_img_cap
                #
                # # SI-Q
                # train_text_features_new = train_text_features[index[:32]]
                # # train_text_features_new = torch.tensor(train_text_features_new)
                # similarity_img_cap = (100.0 * test_text_features[idx] @ train_text_features_new.T).softmax(dim=-1)
                # # values_img_cap, indices_img_cap = similarity_img_cap[0].topk(32)
                #
                # values_img_cap_, indices_img_cap_ = similarity_img_cap.topk(32)
                # indices_img_cap = index[indices_img_cap_.tolist()]
                # output_img_cap = []
                # for id, val in zip(indices_img_cap, values_img_cap_):
                #     output_img_cap.append([train_ids[id], val.tolist(), id])
                # output_all[test_ids[idx]]["SI_Q"] = output_img_cap

        def SQ_retrieval():
            index = faiss.IndexFlatL2(512)
            index.add(train_text_features.cpu())
            print(index.ntotal)
            if not os.path.exists('/hy-tmp/VQA_VQAv2/caption_caption_indices_caps.npy'):
                values_caps, indices_caps = index.search(test_text_features.cpu(), 2000)
                np.save("/hy-tmp/VQA_VQAv2/caption_caption_indices_caps.npy", indices_caps)
                np.save("/hy-tmp/VQA_VQAv2/caption_caption_values_caps.npy", values_caps)
            else:
                values_caps = np.load("/hy-tmp/VQA_VQAv2/caption_caption_values_caps.npy")
                indices_caps = np.load("/hy-tmp/VQA_VQAv2/caption_caption_indices_caps.npy")

            # SQ
            for idx, (value, index) in enumerate(zip(values_caps, indices_caps)):
                output_caps = []
                for id, val in zip(index[:32], value[:32]):
                    output_caps.append([train_ids[id], val.tolist(), id])
                output_all[test_ids[idx]]["SQ"] = output_caps

                # candidate_caps = defaultdict(list)
                # for id, val in zip(index, value):
                #     cap_id = str(train_ids[id])[:6]  # str
                #     candidate_caps[cap_id].append([train_ids[id], val.tolist(), id])
                #     if len(candidate_caps) >= 32:
                #         break
                # train_image_features_new_idx = []
                # for candidate_cap in candidate_caps.values():
                #     train_image_features_new_idx += [feature[2] for feature in candidate_cap]
                # train_image_features_new = train_image_features[train_image_features_new_idx]
                # similarity_cap_img = (100.0 * test_image_features[idx] @ train_image_features_new.T).softmax(dim=-1)
                # # values_img_cap, indices_img_cap = similarity_img_cap[0].topk(32)
                # values_cap_img_1, indices_cap_img_1 = similarity_cap_img.topk(len(similarity_cap_img))
                # values_cap_img_, indices_cap_img_ = similarity_cap_img.topk(32)
                # indices_cap_img = index[indices_cap_img_.tolist()]
                # indices_cap_img_1 = index[indices_cap_img_1.tolist()]
                # output_cap_img = []
                # for id, val in zip(indices_cap_img, values_cap_img_):
                #     output_cap_img.append([train_ids[id], val.tolist(), id])
                # output_all[test_ids[idx]]["caption_image_new"] = output_cap_img
                # output_cap_img = []
                # all_ready_in = []
                # for id, val in zip(indices_cap_img_1, values_cap_img_1):
                #     if str(train_ids[id])[:6] not in all_ready_in:
                #         output_cap_img.append([train_ids[id], val.tolist(), id])
                #         all_ready_in.append(str(train_ids[id])[:6])
                #     if len(all_ready_in) >= 32:
                #         break
                # output_all[test_ids[idx]]["caption_image_new_2"] = output_cap_img
                #
                #
                # train_image_features_new = train_image_features[index[:32]]
                # # train_text_features_new = torch.tensor(train_text_features_new)
                # similarity_cap_img = (100.0 * test_image_features[idx] @ train_image_features_new.T).softmax(dim=-1)
                # # values_img_cap, indices_img_cap = similarity_img_cap[0].topk(32)
                # values_cap_img_, indices_cap_img_ = similarity_cap_img.topk(32)
                # indices_cap_img = index[indices_cap_img_.tolist()]
                # output_cap_img = []
                # for id, val in zip(indices_cap_img, values_cap_img_):
                #     output_cap_img.append([train_ids[id], val.tolist(), id])
                # output_all[test_ids[idx]]["caption_image"] = output_cap_img


        def SQA_retrieval():
            index = faiss.IndexFlatL2(512)
            index.add(train_text_features_qa.cpu())
            print(index.ntotal)
            if not os.path.exists('/hy-tmp/VQA_VQAv2/caption_caption_indices_caps_qa.npy'):
                print("Doing SQA search...")
                values_caps, indices_caps = index.search(test_text_features_qa.cpu(), 100)
                np.save("/hy-tmp/VQA_VQAv2/caption_caption_indices_caps_qa.npy", indices_caps)
                np.save("/hy-tmp/VQA_VQAv2/caption_caption_values_caps_qa.npy", values_caps)
                print("Done.")
            else:
                values_caps = np.load("/hy-tmp/VQA_VQAv2/caption_caption_values_caps_qa.npy")
                indices_caps = np.load("/hy-tmp/VQA_VQAv2/caption_caption_indices_caps_qa.npy")

            # SQQR
            for idx, (value, index) in enumerate(zip(values_caps, indices_caps)):
                output_caps = []
                for id, val in zip(index[:32], value[:32]):
                    output_caps.append([train_ids[id], val.tolist(), id])
                output_all[test_ids[idx]]["SQA"] = output_caps


        def I_SQ_retrieval():   #实现从图像到文本的检索功能
            index = faiss.IndexFlatL2(512)
            index.add(train_text_features.cpu())
            print(index.ntotal)
            # I_SQ
            if not os.path.exists('/hy-tmp/VQA_VQAv2/image_question_indices_imgs.npy'):
                print("Doing I_SQ search...")
                values_imgs, indices_imgs = index.search(test_image_features.cpu(), 100)
                np.save("/hy-tmp/VQA_VQAv2/image_question_indices_imgs.npy", indices_imgs)
                np.save("/hy-tmp/VQA_VQAv2/image_question_values_imgs.npy", values_imgs)
                print("Done.")
            else:
                values_imgs = np.load("/hy-tmp/VQA_VQAv2/image_question_values_imgs.npy")
                indices_imgs = np.load("/hy-tmp/VQA_VQAv2/image_question_indices_imgs.npy")

            for idx, (value, index) in enumerate(zip(values_imgs, indices_imgs)):
                output_imgs = []
                for id, val in zip(index[:32], value[:32]):
                    output_imgs.append([train_ids[id], val.tolist(), id])
                output_all[test_ids[idx]]["I_SQ"] = output_imgs

        def I_SQA_retrieval():  #从图像到问题-答案对（Image-to-Question-Answer，简称 I_SQA）的检索功能
            index = faiss.IndexFlatL2(512)
            index.add(train_text_features_qa.cpu())
            print(index.ntotal)
            # I_SQA
            if not os.path.exists('/hy-tmp/VQA_VQAv2/image_question_indices_imgs_qa.npy'):
                print("Doing I_SQA search...")
                values_imgs, indices_imgs = index.search(test_image_features.cpu(), 100)
                np.save("/hy-tmp/VQA_VQAv2/image_question_indices_imgs_qa.npy", indices_imgs)
                np.save("/hy-tmp/VQA_VQAv2/image_question_values_imgs_qa.npy", values_imgs)
                print("Done.")
            else:
                values_imgs = np.load("/hy-tmp/VQA_VQAv2/image_question_values_imgs_qa.npy")
                indices_imgs = np.load("/hy-tmp/VQA_VQAv2/image_question_indices_imgs_qa.npy")

            for idx, (value, index) in enumerate(zip(values_imgs, indices_imgs)):
                output_imgs = []
                for id, val in zip(index[:32], value[:32]):
                    output_imgs.append([train_ids[id], val.tolist(), id])
                output_all[test_ids[idx]]["I_SQA"] = output_imgs

        def Q_SI_retrieval():  #从文本（问题）到图像的检索功能
            index = faiss.IndexFlatL2(512)
            index.add(train_image_features.cpu())
            print(index.ntotal)
            # Q_SI
            if not os.path.exists('/hy-tmp/VQA_VQAv2/question_image_indices_imgs.npy'):
                print("Doing Q_SI search...")
                values_imgs, indices_imgs = index.search(test_text_features.cpu(), 100)
                np.save("/hy-tmp/VQA_VQAv2/question_image_indices_imgs.npy", indices_imgs)
                np.save("/hy-tmp/VQA_VQAv2/question_image_values_imgs.npy", values_imgs)
                print("Done.")
            else:
                values_imgs = np.load("/hy-tmp/VQA_VQAv2/question_image_values_imgs.npy")
                indices_imgs = np.load("/hy-tmp/VQA_VQAv2/question_image_indices_imgs.npy")

            for idx, (value, index) in enumerate(zip(values_imgs, indices_imgs)):
                output_imgs = []
                for id, val in zip(index[:32], value[:32]):
                    output_imgs.append([train_ids[id], val.tolist(), id])
                output_all[test_ids[idx]]["Q_SI"] = output_imgs

        def QA_SI_retrieval():  #通过测试集的问题-答案对的文本特征，检索出与之最相似的训练集图像特征
            index = faiss.IndexFlatL2(512)
            index.add(train_image_features.cpu())
            print(index.ntotal)
            # QA_SI
            if not os.path.exists('/hy-tmp/VQA_VQAv2/question_image_indices_imgs_qa.npy'):
                print("Doing QA_SI search...")
                values_imgs, indices_imgs = index.search(test_text_features_qa.cpu(), 100)
                np.save("/hy-tmp/VQA_VQAv2/question_image_indices_imgs_qa.npy", indices_imgs)
                np.save("/hy-tmp/VQA_VQAv2/question_image_values_imgs_qa.npy", values_imgs)
                print("Done.")
            else:
                values_imgs = np.load("/hy-tmp/VQA_VQAv2/question_image_values_imgs_qa.npy")
                indices_imgs = np.load("/hy-tmp/VQA_VQAv2/question_image_indices_imgs_qa.npy")

            for idx, (value, index) in enumerate(zip(values_imgs, indices_imgs)):
                output_imgs = []
                for id, val in zip(index[:32], value[:32]):
                    output_imgs.append([train_ids[id], val.tolist(), id])
                output_all[test_ids[idx]]["QA_SI"] = output_imgs

        def SmixR7_retrieval():  #混合检索功能，称为 SmixR（Mixed Retrieval）。
                                #它结合了图像到图像（SI）和文本到文本（SQ）的检索结果
            print("Doing SmixR search...")

            index = faiss.IndexFlatL2(512)
            index.add(train_image_features.cpu())
            print(index.ntotal)
            # SI
            if not os.path.exists('/hy-tmp/cp_npy/image_image_indices_imgs.npy'):
                print("First: Doing SI search...")
                values_imgs, indices_imgs = index.search(test_image_features.cpu(), 10000)
                np.save("/hy-tmp/cp_npy/image_image_indices_imgs.npy", indices_imgs)
                np.save("/hy-tmp/cp_npy/image_image_values_imgs.npy", values_imgs)
                print("Done.")
            else:
                values_imgs = np.load("/hy-tmp/cp_npy/image_image_values_imgs.npy")
                indices_imgs = np.load("/hy-tmp/cp_npy/image_image_indices_imgs.npy")

            # if not os.path.exists('image_image_indices_imgs.npy'):
            #     values_imgs, indices_imgs = index.search(test_image_features.cpu(), 2000)
            #     np.save("image_image_indices_imgs.npy", indices_imgs)
            #     np.save("image_image_values_imgs.npy", values_imgs)
            # else:
            #     values_imgs = np.load("image_image_values_imgs.npy")
            #     indices_imgs = np.load("image_image_indices_imgs.npy")

            # SQ
            index = faiss.IndexFlatL2(512)
            index.add(train_text_features.cpu())
            print(index.ntotal)
            if not os.path.exists('/hy-tmp/cp_npy/caption_caption_indices_caps.npy'):
                print("Second: Doing SQ search...")
                values_caps, indices_caps = index.search(test_text_features.cpu(), 10000)
                np.save("/hy-tmp/cp_npy/caption_caption_indices_caps.npy", indices_caps)
                np.save("/hy-tmp/cp_npy/caption_caption_values_caps.npy", values_caps)
                print("Done.")
            else:
                values_caps = np.load("/hy-tmp/cp_npy/caption_caption_values_caps.npy")
                indices_caps = np.load("/hy-tmp/cp_npy/caption_caption_indices_caps.npy")

            # if not os.path.exists('caption_caption_indices_caps.npy'):
            #     values_caps, indices_caps = index.search(test_text_features.cpu(), 2000)
            #     np.save("caption_caption_indices_caps.npy", indices_caps)
            #     np.save("caption_caption_values_caps.npy", values_caps)
            # else:
            #     values_caps = np.load("caption_caption_values_caps.npy")
            #     indices_caps = np.load("caption_caption_indices_caps.npy")

            for idx, (value_img, index_img, value_cap, index_cap) \
                    in enumerate(zip(values_imgs, indices_imgs, values_caps, indices_caps)):
                output_mix = []
                mix_score = defaultdict(int)
                visited = defaultdict(int)
                for id, val in zip(index_img, value_img):
                    mix_score[id] += 0.7 * val
                    visited[id] += 1
                for id, val in zip(index_cap, value_cap):
                    mix_score[id] += 0.3 * val
                    visited[id] += 1

                sorted_mix_score = sorted(mix_score.items(), key=lambda i: i[1], reverse=True)
                for result in sorted_mix_score[:32]:
                    output_mix.append([train_ids[result[0]], result[1].tolist(), result[0]])
                output_all[test_ids[idx]]["SmixR7"] = output_mix
                
        #通过测试集的问题和生成的答案的组合特征，检索出与之最相似的训练集问题-答案对的文本特征
        def SQA_generated_retrieval():
            signal = "SI_4(1)"
            # load result file
            result_file = "/hy-tmp/Iteration/iter0/vqav2results_VQAv2_Result_file_name_8-shot.json"  #加载生成的答案文件 VQAv2results_SI_4-shot.json
            generated_set = json.load(open(result_file, "r"))
            generated_answers = {}
            for item in generated_set:
                generated_answers[item["question_id"]] = item["answer"]

            # get qa(generated) and extract the feature
            #使用 CLIP 的 tokenize 方法将测试集的问题和生成的答案组合成输入文本
            text_features = []
            if not os.path.exists('/hy-tmp/Iteration/iter1/test_text_features_qa_{}generated.npy'.format(signal)):
                print("Extract test qa({} generated)...".format(signal))
                #filtered_test_captions = [i for i in test_captions if i["question_id"] in generated_answers]
                text_inputs = clip.tokenize(
                    [i['question'] + generated_answers[i["question_id"]]
                     if i["question_id"] in generated_answers.keys() else "default"
                     for i in test_captions])
                for i in tqdm(text_inputs):
                    with torch.no_grad():
                        text_features.append(model.encode_text(i.unsqueeze(0).to(device)))
                text_features = torch.stack(text_features).squeeze(1)
                np.save('/hy-tmp/Iteration/iter1/test_text_features_qa_{}generated.npy'.format(signal), text_features.to("cpu"))
                print("Done test qa.")

            test_text_features_qa_gr = torch.from_numpy(
                np.load('/hy-tmp/Iteration/iter1/test_text_features_qa_{}generated.npy'.format(signal))).to(device)
            print(test_text_features_qa_gr.shape)
            test_text_features_qa_gr /= test_text_features_qa_gr.norm(dim=-1, keepdim=True)

            # do sqa
            index = faiss.IndexFlatL2(512)
            index.add(train_text_features_qa.cpu())
            print(index.ntotal)
            if not os.path.exists('/hy-tmp/Iteration/iter1/caption_caption_indices_caps_qa_{}generated.npy'.format(signal)):
                print("Doing SQA search...")
                values_caps, indices_caps = index.search(test_text_features_qa_gr.cpu(), 100)
                np.save('/hy-tmp/Iteration/iter1/caption_caption_indices_caps_qa_{}generated.npy'.format(signal), indices_caps)
                np.save('/hy-tmp/Iteration/iter1/caption_caption_values_caps_qa_{}generated.npy'.format(signal), values_caps)
                print("Done.")
            else:
                values_caps = np.load('/hy-tmp/Iteration/iter1/caption_caption_values_caps_qa_{}generated.npy'.format(signal))
                indices_caps = np.load('/hy-tmp/Iteration/iter1/caption_caption_indices_caps_qa_{}generated.npy'.format(signal))

            # SQQR
            for idx, (value, index) in enumerate(zip(values_caps, indices_caps)):
                output_caps = []
                for id, val in zip(index[:32], value[:32]):
                    output_caps.append([train_ids[id], val.tolist(), id])
                output_all[test_ids[idx]]["SQA_{}".format(signal)] = output_caps
        def RS_retrieval():
            # 确保随机选择的结果可复现
            random.seed(42)

            # 检查是否已经保存了随机选择的结果
            if not os.path.exists('/hy-tmp/cp_npy/image_image_indices_imgs_random.npy'):
                # 随机选择样本
                indices_imgs = []
                for _ in range(len(test_image_features)):
                    # 从训练集中随机选择32个样本
                    random_indices = random.sample(range(len(train_ids)), 32)
                    indices_imgs.append(random_indices)
                indices_imgs = np.array(indices_imgs)
                np.save("/hy-tmp/cp_npy/image_image_indices_imgs_random.npy", indices_imgs)
            else:
                indices_imgs = np.load("/hy-tmp/cp_npy/image_image_indices_imgs_random.npy")

            # 生成输出
            for idx, index in enumerate(indices_imgs):
                output_imgs = []
                for id in index[:32]:
                    output_imgs.append([train_ids[id], 0.0, id])  # 随机选择不涉及相似性分数，因此分数设为0.0
                output_all[test_ids[idx]]["RS"] = output_imgs


        preprocess_signals = {
            "SI" : False,
            "SQ" : False,
            "I_SQ" : False,
            "Q_SI" : False,
            "SQA" : False,
            "I_SQA" : False,
            "QA_SI" : False,
            "SmixR7" : False,
            "SQA_generated": False,
            "RS" : True,
        }

        for signal in preprocess_signals.keys():
            if preprocess_signals[signal]:
                locals()[signal + "_retrieval"]()
                np.save('/hy-tmp/VQA_VQAv2/retrieval/retrieval_results/vqa/validation_VQAv2_{}.npy'.format(signal), output_all)

    def diversity_retrieval():
        output_all = defaultdict(dict)

        diir_path = "SGG_tags/test_tags_topn_DIIR_TR/"

        already = []
        output_rs = []
        for idx, test_img in tqdm(enumerate(test_img_ids)):
            if test_img in already:
                output_all[test_ids[idx]]["DIIR_TR_rs"] = output_rs
                continue
            output_rs = []
            try:
                sim_json = json.load(open(os.path.join(diir_path, str(test_img) + "_sim.json"), "r"))
            except:
                continue
            for i in sim_json:
                candidate_trains = []
                for id, question in enumerate(train_captions):
                    if question['image_id'] == i[0]:
                        candidate_trains.append(id)
                        # output_rs.append([train_ids[id], i[1], id])
                        # rs has repetitive images, need a single version
                train_text_features_new = train_text_features[candidate_trains]
                similarity_img_cap = (100.0 * test_text_features[idx] @ train_text_features_new.T).softmax(dim=-1)
                values_img_cap, indices_img_cap = similarity_img_cap.topk(1)
                indices_img_cap = [candidate_trains[i] for i in indices_img_cap.tolist()]
                for id, val in zip(indices_img_cap, values_img_cap):
                    output_rs.append([train_ids[id], val.tolist(), id])
                if len(output_rs) >= 32:
                    break
            already.append(test_img)
            output_all[test_ids[idx]]["DIIR_TR_rs"] = output_rs
        np.save('retrieval_results/vqa/validation_di_single_tr.npy', output_all)

    similarity_retrieval()
    print("debug")


    
########################VizWiz数据集实验########################



def retrieve_vqa_vizwiz():
    train_data_dir = "/hy-tmp/VQA_VQAv2/retrieval/vizwiz/image/train/"
    val_data_dir = "/hy-tmp/VQA_VQAv2/retrieval/vizwiz/image/val/"

    # prepare train data
    train_captions = json.load(open("/hy-tmp/VQA_VQAv2/retrieval/vizwiz/train_questions_vqa_format.json", 'r'))["questions"]
    train_answers = json.load(open("/hy-tmp/VQA_VQAv2/retrieval/vizwiz/train_annotations_vqa_format.json", 'r'))["annotations"]
    # prepare test data
    test_captions = json.load(open("/hy-tmp/VQA_VQAv2/retrieval/vizwiz/val_questions_vqa_format.json", 'r'))["questions"]
    test_answers = json.load(open("/hy-tmp/VQA_VQAv2/retrieval/vizwiz/val_annotations_vqa_format.json", 'r'))["annotations"]
    # extract train feature
    train_ids = []
    train_img_ids = []
    train_filenames = []
    for idx, caption in enumerate(train_captions):
        train_ids.append(caption["question_id"])
        train_img_ids.append(caption["image_id"])
        train_name = str(caption["image_id"])
        train_filenames.append(os.path.join(train_data_dir, train_name))

    if not os.path.exists('train_image_features_vizwiz.npy'):
        train_features = extract_features(train_filenames)
        np.save("train_image_features_vizwiz.npy", train_features.to("cpu"))

    # text encode
    text_features = []
    if not os.path.exists('train_text_features_vizwiz.npy'):
        text_inputs = clip.tokenize([i['question'] for i in train_captions])
        for i in tqdm(text_inputs):
            with torch.no_grad():
                text_features.append(model.encode_text(i.unsqueeze(0).to(device)))
        text_features = torch.stack(text_features).squeeze(1)
        np.save("train_text_features_vizwiz.npy", text_features.to("cpu"))
    # qa
    text_features = []
    if not os.path.exists('train_text_features_qa_vizwiz.npy'):
        print("Extract train qa...")
        text_inputs = clip.tokenize([i['question'] + j['multiple_choice_answer']
                                     for (i, j) in zip(train_captions, train_answers)],truncate=True) #clip.tokenize最长只能接受77个标记，再长就截断它
        for i in tqdm(text_inputs):
            with torch.no_grad():
                text_features.append(model.encode_text(i.unsqueeze(0).to(device)))
        text_features = torch.stack(text_features).squeeze(1)
        np.save("train_text_features_qa_vizwiz.npy", text_features.to("cpu"))
        print("Done train qa.")

    # read features
    train_image_features = torch.from_numpy(np.load("train_image_features_vizwiz.npy")).to(device)
    train_text_features = torch.from_numpy(np.load("train_text_features_vizwiz.npy")).to(device)
    train_text_features_qa = torch.from_numpy(np.load("train_text_features_qa_vizwiz.npy")).to(device)
    print(train_image_features.shape, train_text_features.shape, train_text_features_qa.shape)

    # extract test feature
    test_ids = []
    test_img_ids = []
    test_filenames = []
    for caption in test_captions:
        test_ids.append(caption["question_id"])
        test_img_ids.append(caption["image_id"])
        test_name = str(caption["image_id"])
        test_filenames.append(os.path.join(val_data_dir, test_name))
    if not os.path.exists('test_image_features_vizwiz.npy'):
        test_features = extract_features(test_filenames)
        np.save("test_image_features_vizwiz.npy", test_features.to("cpu"))

    # text encode
    text_features = []
    if not os.path.exists('test_text_features_vizwiz.npy'):
        text_inputs = clip.tokenize([i['question'] for i in test_captions])
        for i in tqdm(text_inputs):
            with torch.no_grad():
                text_features.append(model.encode_text(i.unsqueeze(0).to(device)))
        text_features = torch.stack(text_features).squeeze(1)
        np.save("test_text_features_vizwiz.npy", text_features.to("cpu"))

    text_features = []
    if not os.path.exists('test_text_features_qa_vizwiz.npy'):
        print("Extract test qa...")
        text_inputs = clip.tokenize([i['question'] + j['multiple_choice_answer'] for (i, j) in zip(test_captions, test_answers)],truncate=True)
        for i in tqdm(text_inputs):
            with torch.no_grad():
                text_features.append(model.encode_text(i.unsqueeze(0).to(device)))
        text_features = torch.stack(text_features).squeeze(1)
        np.save("test_text_features_qa_vizwiz.npy", text_features.to("cpu"))
        print("Done test qa.")

    # read features
    test_image_features = torch.from_numpy(np.load("test_image_features_vizwiz.npy")).to(device)
    test_text_features = torch.from_numpy(np.load("test_text_features_vizwiz.npy")).to(device)
    test_text_features_qa = torch.from_numpy(np.load("test_text_features_qa_vizwiz.npy")).to(device)
    print(test_image_features.shape, test_text_features.shape, test_text_features_qa.shape)

    # softmax
    test_image_features /= test_image_features.norm(dim=-1, keepdim=True)
    test_text_features /= test_text_features.norm(dim=-1, keepdim=True)
    test_text_features_qa /= test_text_features_qa.norm(dim=-1, keepdim=True)
    train_image_features /= train_image_features.norm(dim=-1, keepdim=True)
    train_text_features /= train_text_features.norm(dim=-1, keepdim=True)
    train_text_features_qa /= train_text_features_qa.norm(dim=-1, keepdim=True)


    output_all = defaultdict(dict)

    def SI_retrieval():
        index = faiss.IndexFlatL2(512)
        index.add(train_image_features.cpu())
        print(index.ntotal)
        # SIdf
        if not os.path.exists('image_image_indices_imgs_vizwiz.npy'):
            print("Doing SI search...")
            values_imgs, indices_imgs = index.search(test_image_features.cpu(), 2000)
            np.save("image_image_indices_imgs_vizwiz.npy", indices_imgs)
            np.save("image_image_values_imgs_vizwiz.npy", values_imgs)
            print("Done.")
        else:
            values_imgs = np.load("image_image_values_imgs_vizwiz.npy")
            indices_imgs = np.load("image_image_indices_imgs_vizwiz.npy")
        for idx, (value, index) in enumerate(zip(values_imgs, indices_imgs)):
            output_imgs = []
            output_imgs_new = []
            for id, val in zip(index[:32], value[:32]):
                output_imgs.append([train_ids[id], val.tolist(), id])
            output_all[test_ids[idx]]["SI"] = output_imgs
            # SI+_1
            candidate_imgs = defaultdict(list)
            for id, val in zip(index, value):
                img_id = str(train_ids[id])[:6]  # str
                candidate_imgs[img_id].append([train_ids[id], val.tolist(), id])
                if len(candidate_imgs) >= 32:
                    break
            train_text_features_new_idx = []
            for candidate_img in candidate_imgs.values():
                train_text_features_new_idx += [feature[2] for feature in candidate_img]
            train_text_features_new = train_text_features[train_text_features_new_idx]
            similarity_img_cap = (100.0 * test_text_features[idx] @ train_text_features_new.T).softmax(dim=-1)
            # values_img_cap, indices_img_cap = similarity_img_cap[0].topk(32)
            values_img_cap_1, indices_img_cap_1 = similarity_img_cap.topk(len(similarity_img_cap))
            values_img_cap_, indices_img_cap_ = similarity_img_cap.topk(32)
            indices_img_cap = index[indices_img_cap_.tolist()]
            indices_img_cap_1 = index[indices_img_cap_1.tolist()]
            output_img_cap = []
            for id, val in zip(indices_img_cap, values_img_cap_):
                output_img_cap.append([train_ids[id], val.tolist(), id])
            output_all[test_ids[idx]]["SI_1"] = output_img_cap
            # SI+_2
            output_img_cap = []
            all_ready_in = []
            for id, val in zip(indices_img_cap_1, values_img_cap_1):
                if str(train_ids[id])[:6] not in all_ready_in:
                    output_img_cap.append([train_ids[id], val.tolist(), id])
                    all_ready_in.append(str(train_ids[id])[:6])
                if len(all_ready_in) >= 32:
                    break
            output_all[test_ids[idx]]["SI_2"] = output_img_cap

            # SI-Q
            train_text_features_new = train_text_features[index[:32]]
            # train_text_features_new = torch.tensor(train_text_features_new)
            similarity_img_cap = (100.0 * test_text_features[idx] @ train_text_features_new.T).softmax(dim=-1)
            # values_img_cap, indices_img_cap = similarity_img_cap[0].topk(32)

            values_img_cap_, indices_img_cap_ = similarity_img_cap.topk(32)
            indices_img_cap = index[indices_img_cap_.tolist()]
            output_img_cap = []
            for id, val in zip(indices_img_cap, values_img_cap_):
                output_img_cap.append([train_ids[id], val.tolist(), id])
            output_all[test_ids[idx]]["SI_Q"] = output_img_cap

    def SQ_retrieval():
        index = faiss.IndexFlatL2(512)
        index.add(train_text_features.cpu())
        print(index.ntotal)
        if not os.path.exists('caption_caption_indices_caps_vizwiz.npy'):
            print("Doing SQ search...")
            values_caps, indices_caps = index.search(test_text_features.cpu(), 2000)
            np.save("caption_caption_indices_caps_vizwiz.npy", indices_caps)
            np.save("caption_caption_values_caps_vizwiz.npy", values_caps)
            print("Done.")
        else:
            values_caps = np.load("caption_caption_values_caps_vizwiz.npy")
            indices_caps = np.load("caption_caption_indices_caps_vizwiz.npy")

        # SQ
        for idx, (value, index) in enumerate(zip(values_caps, indices_caps)):
            output_caps = []
            for id, val in zip(index[:32], value[:32]):
                output_caps.append([train_ids[id], val.tolist(), id])
            output_all[test_ids[idx]]["SQ"] = output_caps
            train_image_features_new = train_image_features[index[:32]]
            # train_text_features_new = torch.tensor(train_text_features_new)
            similarity_cap_img = (100.0 * test_image_features[idx] @ train_image_features_new.T).softmax(dim=-1)
            # values_img_cap, indices_img_cap = similarity_img_cap[0].topk(32)
            values_cap_img_, indices_cap_img_ = similarity_cap_img.topk(32)
            indices_cap_img = index[indices_cap_img_.tolist()]
            output_cap_img = []
            for id, val in zip(indices_cap_img, values_cap_img_):
                output_cap_img.append([train_ids[id], val.tolist(), id])
            output_all[test_ids[idx]]["SQ_I"] = output_cap_img

    def SQA_retrieval():
        index = faiss.IndexFlatL2(512)
        index.add(train_text_features_qa.cpu())
        print(index.ntotal)
        if not os.path.exists('caption_caption_indices_caps_qa_vizwiz.npy'):
            print("Doing SQA search...")
            values_caps, indices_caps = index.search(test_text_features_qa.cpu(), 100)
            np.save("caption_caption_indices_caps_qa_vizwiz.npy", indices_caps)
            np.save("caption_caption_values_caps_qa_vizwiz.npy", values_caps)
            print("Done.")
        else:
            values_caps = np.load("caption_caption_values_caps_qa_vizwiz.npy")
            indices_caps = np.load("caption_caption_indices_caps_qa_vizwiz.npy")

        # SQQR
        for idx, (value, index) in enumerate(zip(values_caps, indices_caps)):
            output_caps = []
            for id, val in zip(index[:32], value[:32]):
                output_caps.append([train_ids[id], val.tolist(), id])
            output_all[test_ids[idx]]["SQA"] = output_caps

    def SQA_generated_retrieval():
        signal = "RS_v1_4"
        # load result file
        result_file = "vizwiz_result_RS_4_shot.json"
        generated_set = json.load(open(result_file, "r"))
        generated_answers = {}
        for item in generated_set:
            generated_answers[item["question_id"]] = item["answer"]

        # get qa(generated) and extract the feature
        text_features = []
        if not os.path.exists('test_text_features_qa_{}generated_vizwiz.npy'.format(signal)):
            print("Extract test qa({} generated)...".format(signal))
            text_inputs = clip.tokenize(
                [i['question'] + generated_answers[i["question_id"]]
                 if i["question_id"] in generated_answers.keys() else "default"
                 for i in test_captions])
            for i in tqdm(text_inputs):
                with torch.no_grad():
                    text_features.append(model.encode_text(i.unsqueeze(0).to(device)))
            text_features = torch.stack(text_features).squeeze(1)
            np.save('test_text_features_qa_{}generated_vizwiz.npy'.format(signal), text_features.to("cpu"))
            print("Done test qa.")

        test_text_features_qa_gr = torch.from_numpy(
            np.load('test_text_features_qa_{}generated_vizwiz.npy'.format(signal))).to(device)
        print(test_text_features_qa_gr.shape)
        test_text_features_qa_gr /= test_text_features_qa_gr.norm(dim=-1, keepdim=True)

        # do sqa
        index = faiss.IndexFlatL2(512)
        index.add(train_text_features_qa.cpu())
        print(index.ntotal)
        if not os.path.exists('caption_caption_indices_caps_qa_{}generated_vizwiz.npy'.format(signal)):
            print("Doing SQAQAR search...")
            values_caps, indices_caps = index.search(test_text_features_qa_gr.cpu(), 100)
            np.save('caption_caption_indices_caps_qa_{}generated_vizwiz.npy'.format(signal), indices_caps)
            np.save('caption_caption_values_caps_qa_{}generated_vizwiz.npy'.format(signal), values_caps)
            print("Done.")
        else:
            values_caps = np.load('caption_caption_values_caps_qa_{}generated_vizwiz.npy'.format(signal))
            indices_caps = np.load('caption_caption_indices_caps_qa_{}generated_vizwiz.npy'.format(signal))

        # SQ
        for idx, (value, index) in enumerate(zip(values_caps, indices_caps)):
            output_caps = []
            for id, val in zip(index[:32], value[:32]):
                output_caps.append([train_ids[id], val.tolist(), id])
            output_all[test_ids[idx]]["SQA_{}".format(signal)] = output_caps

    preprocess_signals = {
        "SI": False,
        "SQ": False,
        # "I_SQ": False,
        # "Q_SI": False,
        "SQA": True,
        # "I_SQA": False,
        # "QA_SI": False,
        # "SmixR": False,
        # "SQA_generated": True,
    }
    for signal in preprocess_signals.keys():
        if preprocess_signals[signal]:
            locals()[signal + "_retrieval"]()
            np.save('/hy-tmp/VQA_VQAv2/retrieval/retrieval_results/vizwiz_validation_{}.npy'.format(signal), output_all)



########################okvqa数据集实验########################



def retrieve_vqa_okvqa():
    train_data_dir = "/hy-tmp/VQA_VQAv2/retrieval/mscoco2014/train2014/"
    val_data_dir = "/hy-tmp/VQA_VQAv2/retrieval/mscoco2014/val2014"

    # prepare train data
    train_captions = json.load(open("/hy-tmp/VQA_VQAv2/retrieval/okvqa/OpenEnded_mscoco_train2014_questions.json", 'r'))["questions"]
    train_answers = json.load(open("/hy-tmp/VQA_VQAv2/retrieval/okvqa/mscoco_train2014_annotations.json", 'r'))["annotations"]
    # prepare test data
    test_captions = json.load(open("/hy-tmp/VQA_VQAv2/retrieval/okvqa/OpenEnded_mscoco_val2014_questions.json", 'r'))["questions"]
    test_answers = json.load(open("/hy-tmp/VQA_VQAv2/retrieval/okvqa/mscoco_val2014_annotations.json", 'r'))["annotations"]

    if not os.path.exists('/hy-tmp/VQA_VQAv2/retrieval/retrieval_results/'):
        os.mkdir('/hy-tmp/VQA_VQAv2/retrieval/retrieval_results/')

    # extract train feature
    train_ids = []
    train_img_ids = []
    train_filenames = []
    for idx, caption in enumerate(train_captions):
        train_ids.append(caption["question_id"])
        train_img_ids.append(caption["image_id"])
        train_name = "COCO_train2014_" + str(caption["image_id"]).zfill(12) + ".jpg"
        train_filenames.append(os.path.join(train_data_dir, train_name))

    if not os.path.exists('train_image_features_okvqa.npy'):
        train_features = extract_features(train_filenames)
        np.save("train_image_features_okvqa.npy", train_features.to("cpu"))

    # text encode
    text_features = []
    if not os.path.exists('train_text_features_okvqa.npy'):
        text_inputs = clip.tokenize([i['question'] for i in train_captions])
        for i in tqdm(text_inputs):
            with torch.no_grad():
                text_features.append(model.encode_text(i.unsqueeze(0).to(device)))
        text_features = torch.stack(text_features).squeeze(1)
        np.save("train_text_features_okvqa.npy", text_features.to("cpu"))
    # qa
    text_features = []
    if not os.path.exists('train_text_features_qa_okvqa.npy'):
        print("Extract train qa...")
        text_inputs = clip.tokenize(
            [i['question'] + j['answers'][0]['answer'] for (i, j) in zip(train_captions, train_answers)])
        for i in tqdm(text_inputs):
            with torch.no_grad():
                text_features.append(model.encode_text(i.unsqueeze(0).to(device)))
        text_features = torch.stack(text_features).squeeze(1)
        np.save("train_text_features_qa_okvqa.npy", text_features.to("cpu"))
        print("Done train qa.")

    # read features
    train_image_features = torch.from_numpy(np.load("train_image_features_okvqa.npy")).to(device)
    train_text_features = torch.from_numpy(np.load("train_text_features_okvqa.npy")).to(device)
    train_text_features_qa = torch.from_numpy(np.load("train_text_features_qa_okvqa.npy")).to(device)
    print(train_image_features.shape, train_text_features.shape, train_text_features_qa.shape)


    # extract test feature
    test_ids = []
    test_img_ids = []
    test_filenames = []
    for caption in test_captions:
        test_ids.append(caption["question_id"])
        test_img_ids.append(caption["image_id"])
        test_name = "COCO_val2014_" + str(caption["image_id"]).zfill(12) + ".jpg"
        test_filenames.append(os.path.join(val_data_dir, test_name))
    if not os.path.exists('test_image_features_okvqa.npy'):
        test_features = extract_features(test_filenames)
        np.save("test_image_features_okvqa.npy", test_features.to("cpu"))

    # text encode
    text_features = []
    if not os.path.exists('test_text_features_okvqa.npy'):
        text_inputs = clip.tokenize([i['question'] for i in test_captions])
        for i in tqdm(text_inputs):
            with torch.no_grad():
                text_features.append(model.encode_text(i.unsqueeze(0).to(device)))
        text_features = torch.stack(text_features).squeeze(1)
        np.save("test_text_features_okvqa.npy", text_features.to("cpu"))

    text_features = []
    if not os.path.exists('test_text_features_qa_okvqa.npy'):
        print("Extract test qa...")
        text_inputs = clip.tokenize(
            [i['question'] + j['answers'][0]['answer'] for (i, j) in zip(test_captions, test_answers)])
        for i in tqdm(text_inputs):
            with torch.no_grad():
                text_features.append(model.encode_text(i.unsqueeze(0).to(device)))
        text_features = torch.stack(text_features).squeeze(1)
        np.save("test_text_features_qa_okvqa.npy", text_features.to("cpu"))
        print("Done test qa.")

    # read features
    test_image_features = torch.from_numpy(np.load("test_image_features_okvqa.npy")).to(device)
    test_text_features = torch.from_numpy(np.load("test_text_features_okvqa.npy")).to(device)
    test_text_features_qa = torch.from_numpy(np.load("test_text_features_qa_okvqa.npy")).to(device)
    print(test_image_features.shape, test_text_features.shape, test_text_features_qa.shape)

    # softmax
    test_image_features /= test_image_features.norm(dim=-1, keepdim=True)
    test_text_features /= test_text_features.norm(dim=-1, keepdim=True)
    test_text_features_qa /= test_text_features_qa.norm(dim=-1, keepdim=True)
    train_image_features /= train_image_features.norm(dim=-1, keepdim=True)
    train_text_features /= train_text_features.norm(dim=-1, keepdim=True)
    train_text_features_qa /= train_text_features_qa.norm(dim=-1, keepdim=True)

    import faiss
    output_all = defaultdict(dict)

    def SI_retrieval():
        index = faiss.IndexFlatL2(512)
        index.add(train_image_features.cpu())
        print(index.ntotal)
        # SI
        if not os.path.exists('image_image_indices_imgs_okvqa.npy'):
            print("Doing SI search...")
            values_imgs, indices_imgs = index.search(test_image_features.cpu(), 2000)
            np.save("image_image_indices_imgs_okvqa.npy", indices_imgs)
            np.save("image_image_values_imgs_okvqa.npy", values_imgs)
            print("Done.")
        else:
            values_imgs = np.load("image_image_values_imgs_okvqa.npy")
            indices_imgs = np.load("image_image_indices_imgs_okvqa.npy")
        for idx, (value, index) in enumerate(zip(values_imgs, indices_imgs)):
            output_imgs = []
            output_imgs_new = []
            for id, val in zip(index[:32], value[:32]):
                output_imgs.append([train_ids[id], val.tolist(), id])
            output_all[test_ids[idx]]["SI"] = output_imgs
            # # SI+_1
            # candidate_imgs = defaultdict(list)
            # for id, val in zip(index, value):
            #     img_id = str(train_ids[id])[:6]  # str
            #     candidate_imgs[img_id].append([train_ids[id], val.tolist(), id])
            #     if len(candidate_imgs) >= 32:
            #         break
            # train_text_features_new_idx = []
            # for candidate_img in candidate_imgs.values():
            #     train_text_features_new_idx += [feature[2] for feature in candidate_img]
            # train_text_features_new = train_text_features[train_text_features_new_idx]
            # similarity_img_cap = (100.0 * test_text_features[idx] @ train_text_features_new.T).softmax(dim=-1)
            # # values_img_cap, indices_img_cap = similarity_img_cap[0].topk(32)
            # values_img_cap_1, indices_img_cap_1 = similarity_img_cap.topk(len(similarity_img_cap))
            # values_img_cap_, indices_img_cap_ = similarity_img_cap.topk(32)
            # indices_img_cap = index[indices_img_cap_.tolist()]
            # indices_img_cap_1 = index[indices_img_cap_1.tolist()]
            # output_img_cap = []
            # for id, val in zip(indices_img_cap, values_img_cap_):
            #     output_img_cap.append([train_ids[id], val.tolist(), id])
            # output_all[test_ids[idx]]["SI_1"] = output_img_cap
            # # SI+_2
            # output_img_cap = []
            # all_ready_in = []
            # for id, val in zip(indices_img_cap_1, values_img_cap_1):
            #     if str(train_ids[id])[:6] not in all_ready_in:
            #         output_img_cap.append([train_ids[id], val.tolist(), id])
            #         all_ready_in.append(str(train_ids[id])[:6])
            #     if len(all_ready_in) >= 32:
            #         break
            # output_all[test_ids[idx]]["SI_2"] = output_img_cap
            #
            # # SI_Q
            # train_text_features_new = train_text_features[index[:32]]
            # # train_text_features_new = torch.tensor(train_text_features_new)
            # similarity_img_cap = (100.0 * test_text_features[idx] @ train_text_features_new.T).softmax(dim=-1)
            # # values_img_cap, indices_img_cap = similarity_img_cap[0].topk(32)
            #
            # values_img_cap_, indices_img_cap_ = similarity_img_cap.topk(32)
            # indices_img_cap = index[indices_img_cap_.tolist()]
            # output_img_cap = []
            # for id, val in zip(indices_img_cap, values_img_cap_):
            #     output_img_cap.append([train_ids[id], val.tolist(), id])
            # output_all[test_ids[idx]]["SI_Q"] = output_img_cap

    def SQ_retrieval():
        index = faiss.IndexFlatL2(512)
        index.add(train_text_features.cpu())
        print(index.ntotal)
        if not os.path.exists('caption_caption_indices_caps_okvqa.npy'):
            print("Doing SQ search...")
            values_caps, indices_caps = index.search(test_text_features.cpu(), 2000)
            np.save("caption_caption_indices_caps_okvqa.npy", indices_caps)
            np.save("caption_caption_values_caps_okvqa.npy", values_caps)
            print("Done.")
        else:
            values_caps = np.load("caption_caption_values_caps_okvqa.npy")
            indices_caps = np.load("caption_caption_indices_caps_okvqa.npy")

        # SQ
        for idx, (value, index) in enumerate(zip(values_caps, indices_caps)):
            output_caps = []
            for id, val in zip(index[:32], value[:32]):
                output_caps.append([train_ids[id], val.tolist(), id])
            output_all[test_ids[idx]]["SQ"] = output_caps
        
            train_image_features_new = train_image_features[index[:32]]
            # train_text_features_new = torch.tensor(train_text_features_new)
            similarity_cap_img = (100.0 * test_image_features[idx] @ train_image_features_new.T).softmax(dim=-1)
            # values_img_cap, indices_img_cap = similarity_img_cap[0].topk(32)
            values_cap_img_, indices_cap_img_ = similarity_cap_img.topk(32)
            indices_cap_img = index[indices_cap_img_.tolist()]
            output_cap_img = []
            for id, val in zip(indices_cap_img, values_cap_img_):
                output_cap_img.append([train_ids[id], val.tolist(), id])
            output_all[test_ids[idx]]["SQ_I"] = output_cap_img

    def SQA_retrieval():
        index = faiss.IndexFlatL2(512)
        index.add(train_text_features_qa.cpu())
        print(index.ntotal)
        if not os.path.exists('caption_caption_indices_caps_qa_okvqa.npy'):
            print("Doing SQA search...")
            values_caps, indices_caps = index.search(test_text_features_qa.cpu(), 100)
            np.save("caption_caption_indices_caps_qa_okvqa.npy", indices_caps)
            np.save("caption_caption_values_caps_qa_okvqa.npy", values_caps)
            print("Done.")
        else:
            values_caps = np.load("caption_caption_values_caps_qa_okvqa.npy")
            indices_caps = np.load("caption_caption_indices_caps_qa_okvqa.npy")

        # SQ
        for idx, (value, index) in enumerate(zip(values_caps, indices_caps)):
            output_caps = []
            for id, val in zip(index[:32], value[:32]):
                output_caps.append([train_ids[id], val.tolist(), id])
            output_all[test_ids[idx]]["SQA"] = output_caps

    def SQA_generated_retrieval():
        signal = "RS_v1_4"
        # load result file
        result_file = "OK_VQAresults_RS_4_shot.json"
        generated_set = json.load(open(result_file, "r"))
        generated_answers = {}
        for item in generated_set:
            generated_answers[item["question_id"]] = item["answer"]

        # get qa(generated) and extract the feature
        text_features = []
        if not os.path.exists('test_text_features_qa_{}generated_okvqa.npy'.format(signal)):
            print("Extract test qa({} generated)...".format(signal))
            text_inputs = clip.tokenize(
                [i['question'] + generated_answers[i["question_id"]]
                 if i["question_id"] in generated_answers.keys() else "default"
                 for i in test_captions])
            for i in tqdm(text_inputs):
                with torch.no_grad():
                    text_features.append(model.encode_text(i.unsqueeze(0).to(device)))
            text_features = torch.stack(text_features).squeeze(1)
            np.save('test_text_features_qa_{}generated_okvqa.npy'.format(signal), text_features.to("cpu"))
            print("Done test qa.")

        test_text_features_qa_gr = torch.from_numpy(
            np.load('test_text_features_qa_{}generated_okvqa.npy'.format(signal))).to(device)
        print(test_text_features_qa_gr.shape)
        test_text_features_qa_gr /= test_text_features_qa_gr.norm(dim=-1, keepdim=True)

        # do sqa
        index = faiss.IndexFlatL2(512)
        index.add(train_text_features_qa.cpu())
        print(index.ntotal)
        if not os.path.exists('caption_caption_indices_caps_qa_{}generated_okvqa.npy'.format(signal)):
            print("Doing SQA search...")
            values_caps, indices_caps = index.search(test_text_features_qa_gr.cpu(), 100)
            np.save('caption_caption_indices_caps_qa_{}generated_okvqa.npy'.format(signal), indices_caps)
            np.save('caption_caption_values_caps_qa_{}generated_okvqa.npy'.format(signal), values_caps)
            print("Done.")
        else:
            values_caps = np.load('caption_caption_values_caps_qa_{}generated_okvqa.npy'.format(signal))
            indices_caps = np.load('caption_caption_indices_caps_qa_{}generated_okvqa.npy'.format(signal))

        # SQQR
        for idx, (value, index) in enumerate(zip(values_caps, indices_caps)):
            output_caps = []
            for id, val in zip(index[:32], value[:32]):
                output_caps.append([train_ids[id], val.tolist(), id])
            output_all[test_ids[idx]]["SQA_{}".format(signal)] = output_caps

    preprocess_signals = {
        # "SI": False,
        "SQ": True,
        # "I_SQ": False,
        # "Q_SI": False,
        # "SQA": False,
        # "I_SQA": False,
        # "QA_SI": False,
        # "SmixR": False,
        # "SQA_generated": True,
    }
    for signal in preprocess_signals.keys():
        if preprocess_signals[signal]:
            locals()[signal + "_retrieval"]()
            np.save('retrieval_results/okvqa_validation_{}.npy'.format(signal), output_all)





if __name__ == "__main__":
    retrieve_vqa_vqav2()
    #retrieve_vqa_okvqa()
    #retrieve_vqa_vizwiz()