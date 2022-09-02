import json
import heapq
import rocketqa

def test_dual_encoder(model, q_file, tp_file):
    dual_encoder = rocketqa.load_model(model=model, use_cuda=True, device_id=0, batch_size=32)

    para_list, title_list, id_list, detail_list = [], [], [], []
    for line in open(tp_file, encoding='utf-8'):
        t, p, id, detail = line.split('\t')
        detail = eval(detail)
        para_list.append(p)
        title_list.append(t)
        id_list.append(id)
        detail_list.append(detail)

    with open('task2.txt', 'w', encoding='utf-8') as fw:
        for line1 in open(q_file, encoding='utf-8'):
            query = line1.strip('\n')
            query_list = [query for _ in range(len(para_list))]

            scores = dual_encoder.matching(query=query_list, \
                                           para=para_list[:len(query_list)], \
                                           title=title_list[:len(query_list)])

            scores = list(scores)
            top5 = heapq.nlargest(5, scores)
            top5index = heapq.nlargest(5, range(len(scores)), scores.__getitem__)

            answer = [{'content-key': id_list[idx], 'detail': detail_list[idx]} for idx in top5index]
            fw.write(json.dumps({
                'question': query,
                'answer': answer
            }, ensure_ascii=False) + '\n')

        fw.close()


if __name__ == '__main__':
    test_dual_encoder('./task2_de/config.json', '../data/test_questions.txt', '../data/task2.para')
