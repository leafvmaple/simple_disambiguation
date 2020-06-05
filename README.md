# 同名消歧-竞赛

该思路代替分为以下几步：

1. 从论文pub中提取出title, keywords, venue, org等features。
2. 将得到的featrues使用word2vec构建embedding，得到一个文本相似矩阵。
3. 基于features使用Heterogeneous network游走，得到的路径使用word2vec构建embedding，得到一个关系相似矩阵。
4. 将相似矩阵和离群论文集送入DBSCAN聚类。
