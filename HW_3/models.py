from scipy.sparse import coo_matrix, csr_matrix
from implicit.als import AlternatingLeastSquares
from typing import List, Union, Dict, Tuple
from collections import Counter
import numpy as np

from scipy.sparse import coo_matrix
from implicit.als import AlternatingLeastSquares
from typing import List, Dict, Tuple, Union
import numpy as np

class ALSModel:
    def __init__(
        self, 
        factors=100, 
        regularization=0.01, 
        alpha=1.0, 
        min_ts: int = 0, 
        max_ts: int = 152,
        fallback_items: List[int] = None
    ):
        self.factors = factors
        self.regularization = regularization
        self.alpha = alpha
        self.min_ts = min_ts
        self.max_ts = max_ts
        self.fallback_items = fallback_items or []

        self.model = AlternatingLeastSquares(factors=factors, regularization=regularization)
        self.user_mapping = {}
        self.item_mapping = {}
        self.user_inv_mapping = {}
        self.item_inv_mapping = {}
        self.trained = False

    def fit(self, df, col='train_interactions'):
        """
        df — DataFrame с колонкой col, в которой лежит список (item_id, timestamp)
        """
        interactions: List[Tuple[int, int, float]] = []

        for _, row in df.iterrows():
            user_id = row['user_id']
            for item_id, ts in row[col]:
                if ts < self.min_ts:
                    continue
                weight = 1 + 0.1 * (ts - self.min_ts)
                interactions.append((user_id, item_id, weight))

        if not interactions:
            raise ValueError("Нет взаимодействий после фильтрации по времени.")

        users, items, weights = zip(*interactions)

        user_ids = list(sorted(set(users)))
        item_ids = list(sorted(set(items)))

        self.user_mapping = {u: i for i, u in enumerate(user_ids)}
        self.item_mapping = {i: j for j, i in enumerate(item_ids)}
        self.user_inv_mapping = {i: u for u, i in self.user_mapping.items()}
        self.item_inv_mapping = {j: i for i, j in self.item_mapping.items()}

        row = [self.user_mapping[u] for u in users]
        col = [self.item_mapping[i] for i in items]

        data = [self.alpha * w for w in weights]
        matrix = coo_matrix((data, (row, col)), shape=(len(user_ids), len(item_ids))).tocsr()

        self.model.fit(matrix)
        self.user_items = matrix
        self.trained = True

    def predict(
        self, 
        df: Union[List[int], 'pd.DataFrame'], 
        topn: int = 10, 
        col: str = 'user_id',
        return_scores: bool = False
    ) -> Dict[int, Union[List[int], List[Tuple[int, float]]]]:
        assert self.trained

        if isinstance(df, list):
            user_ids = df
        else:
            user_ids = df[col].tolist()

        result = {}
        for user_id in user_ids:
            if user_id not in self.user_mapping:
                result[user_id] = (
                    [(item, 0.0) for item in self.fallback_items[:topn]] if return_scores else self.fallback_items[:topn]
                )
                continue

            uid_internal = self.user_mapping[user_id]
            recs, scores = self.model.recommend(
                uid_internal, self.user_items[uid_internal], N=topn
            )
            items = [self.item_inv_mapping[i] for i in recs]
            if return_scores:
                result[user_id] = list(zip(items, scores))
            else:
                result[user_id] = items

        return result


class TopPopular:

    def __init__(self):
        self.trained = False

    def fit(self, df, col='train_interactions'):
        # Подсчёт количества упоминаний каждого item'а
        counter = Counter()
        for row in df[col]:
            for item, _ in row:
                counter[item] += 1

        # Сортировка по популярности
        self.sorted_items_with_counts = counter.most_common()
        self.recommendations = [item for item, _ in self.sorted_items_with_counts]
        self.item_counts = dict(counter)
        self.total_count = sum(counter.values())  # сумма всех взаимодействий
        self.trained = True

    def predict(self, df, topn=10, return_scores: bool = False) -> List[Union[np.ndarray, List[Tuple[int, float]]]]:
        assert self.trained

        top_items = self.recommendations[:topn]

        if return_scores:
            scores = [(item, self.item_counts[item] / self.total_count) for item in top_items]
            return [scores for _ in range(len(df))]

        return [top_items for _ in range(len(df))]


class TopPopularWeighted(TopPopular):

    def __init__(self, min_window=2, max_date=145):
        super().__init__()
        self.min_window = min_window
        self.max_date = max_date

    def fit(self, df, col='train_interactions'):
        from collections import Counter
        counter = Counter()
        
        for row in df[col]:
            for item, timestamp in row:
                if timestamp >= (self.max_date - self.min_window):
                    counter[item] += 1

        self.sorted_items_with_counts = counter.most_common()
        self.recommendations = [item for item, _ in self.sorted_items_with_counts]
        self.item_counts = dict(counter)
        self.total_count = sum(counter.values())
        self.trained = True


def recommend_with_fallback(main_model, fallback_model, dataset, user_ids, k=30):

    # 1. Основные рекомендации от main_model (например, SASRec)
    main_df = main_model.recommend(
        dataset=dataset,
        users=user_ids,
        k=k,
        filter_viewed=True,
        on_unsupported_targets="skip"
    )

    # 2. Кто остался без рекомендаций
    recommended_users = set(main_df["user_id"].unique())
    missing_users = list(set(user_ids) - recommended_users)

    # 3. Fallback-предсказания от эвристической модели
    fallback_recs = fallback_model.predict(pd.DataFrame({"user_id": missing_users}), topn=k, return_scores=True)

    fallback_df = []
    for user, recs in zip(missing_users, fallback_recs):
        for rank, (item_id, score) in enumerate(recs, 1):
            fallback_df.append({
                "user_id": user,
                "item_id": item_id,
                "score": score,
                "rank": rank
            })

    fallback_df = pd.DataFrame(fallback_df)

    # 4. Объединяем обе таблицы
    full_df = pd.concat([main_df, fallback_df], ignore_index=True).sort_values(['user_id', 'rank'])

    return full_df

def ndcg_metric(gt_items, predicted):
    
    at = len(predicted)
    relevance = np.array([1 if x in predicted else 0 for x in gt_items])
    # DCG uses the relevance of the recommended items
    rank_dcg = dcg(relevance)

    if rank_dcg == 0.0:
        return 0.0

    # IDCG has all relevances to 1 (or the values provided), up to the number of items in the test set that can fit in the list length
    ideal_dcg = dcg(np.sort(relevance)[::-1][:at])

    if ideal_dcg == 0.0:
        return 0.0

    ndcg_ = rank_dcg / ideal_dcg

    return ndcg_


def dcg(scores):
    return np.sum(np.divide(np.power(2, scores) - 1, np.log2(np.arange(scores.shape[0], dtype=np.float64) + 2)),
                  dtype=np.float64)


def recall_metric(gt_items, predicted):
    
    n_gt = len(gt_items)
    intersection = len(set(gt_items).intersection(set(predicted)))
    return intersection / min(n_gt, len(gt_items))



def evaluate_recommender(df, model_preds, gt_col='val_interactions', topn=10):
    
    metric_values = []
    
    for idx, row in df.iterrows():
        gt_items = [x[0] for x in row[gt_col]]
        metric_values.append((ndcg_metric(gt_items, row[model_preds]),
                              recall_metric(gt_items, row[model_preds])))
        
    return {'ndcg':float(np.mean([x[0] for x in metric_values])),
            'recall':float(np.mean([x[1] for x in metric_values]))}


