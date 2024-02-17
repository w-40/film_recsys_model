import pymysql
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, Flatten, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
# Connect to MySQL database
db = pymysql.connect(
    host="localhost",
    user="root",
    password="root",
    port=3306,
    database="film_recsys"
)

# Execute SQL query to extract movie data
query = "SELECT id, name, introduction, keywords FROM film"
cursor = db.cursor()
cursor.execute(query)

# Create DataFrame from query results
data = cursor.fetchall()
df = pd.DataFrame(data, columns=['id', 'name', 'introduction', 'keywords'])

# Data preprocessing: Combine text information into one field
# df['combined_text'] = df['introduction'] + ' ' + df['keywords'].apply(lambda x: ' '.join(x.split(',')))
df['combined_text'] = df['keywords'].apply(lambda x: ' '.join(x.split(',')))

# Feature engineering: Use Tokenizer to convert movie name and keywords to sequences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['combined_text'])
sequences = tokenizer.texts_to_sequences(df['combined_text'])

# （之前的导入和数据库连接等部分保持不变）

# 构建 DSSM 模型
max_sequence_length = max(len(seq) for seq in sequences)
vocab_size = len(tokenizer.word_index) + 1

input_layer_query = Input(shape=(max_sequence_length,))
input_layer_candidate = Input(shape=(max_sequence_length,))

embedding_layer_query = Embedding(input_dim=vocab_size, output_dim=50)(input_layer_query)
embedding_layer_candidate = Embedding(input_dim=vocab_size, output_dim=50)(input_layer_candidate)

flatten_layer_query = Flatten()(embedding_layer_query)
flatten_layer_candidate = Flatten()(embedding_layer_candidate)

concatenated_layer = concatenate([flatten_layer_query, flatten_layer_candidate])
dense_layer = Dense(128, activation='relu')(concatenated_layer)
output_layer = Dense(1, activation='sigmoid')(dense_layer)

model = Model(inputs=[input_layer_query, input_layer_candidate], outputs=output_layer)
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# 使用 pad_sequences 填充序列
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post', truncating='post')

# 将数据分为查询和候选集
query_data = padded_sequences
candidate_data = padded_sequences

# 生成用于训练的正样本和负样本
num_samples = len(df)
positive_pairs = []
negative_pairs = []

for i in range(num_samples):
    query_sample = query_data[i]

    # 正样本对（查询，相同电影）
    positive_pairs.append((query_sample, query_sample))

    # 负样本对（查询，不同电影）
    for _ in range(4):
        random_index = np.random.randint(0, num_samples)
        while random_index == i:
            random_index = np.random.randint(0, num_samples)
        negative_pairs.append((query_sample, candidate_data[random_index]))

positive_pairs = np.array(positive_pairs)
negative_pairs = np.array(negative_pairs)

# 合并正样本和负样本用于训练
X_train_query = np.concatenate([positive_pairs[:, 0], negative_pairs[:, 0]])
X_train_candidate = np.concatenate([positive_pairs[:, 1], negative_pairs[:, 1]])
y_train = np.concatenate([np.ones(len(positive_pairs)), np.zeros(len(negative_pairs))])

# 打乱训练数据
indices = np.arange(len(X_train_query))
np.random.shuffle(indices)
X_train_query = X_train_query[indices]
X_train_candidate = X_train_candidate[indices]
y_train = y_train[indices]

# 训练模型
model.fit([X_train_query, X_train_candidate], y_train, epochs=10, batch_size=50)

# 提取电影嵌入
embedding_model = Model(inputs=input_layer_query, outputs=flatten_layer_query)
movie_embeddings = embedding_model.predict(np.array(query_data))

# 计算余弦相似度
cosine_sim = cosine_similarity(movie_embeddings)

# 电影名称到索引的映射
indices = pd.Series(df.index, index=df['name']).drop_duplicates()


def get_recommendations(movie_name, cosine_sim=cosine_sim):
    # 获取电影的索引
    idx = indices[movie_name]

    # 获取电影之间的相似度分数
    sim_scores = list(enumerate(cosine_sim[idx]))

    # 根据相似度分数降序排序
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # 获取前N个相似电影的索引
    top_n_indices = [i[0] for i in sim_scores[1:6]]  # 推荐5部电影

    # 返回推荐电影的名称
    return df['name'].iloc[top_n_indices]


# 示例：给定一个电影名称，获取推荐电影
# while 1:
#     movie_name = input("请输入一个电影名：")
#     recommendations = get_recommendations(movie_name)
#     print(f"推荐给用户观看的电影有：\n{recommendations}")
# 关闭数据库连接
db.close()
