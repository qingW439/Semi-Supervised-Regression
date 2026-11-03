# Tripadvisor 数据清单（Phase 9.1 初步，2026-11-03 20:20）

> 仅做了**无解压的目录级核查**（zip 中央目录 + parquet 元数据 + 首个行组抽样）。
> 尚未做清洗、去重、语言统计与领域适配实验；M1 未完成前不进入正式旅游域实验（任务书 Phase 8 条件）。

## 1. 文件

`D:\tripadvisor_tree_crawler\data\tripadvisor_reviews.zip`，984.97 MB，3 个 parquet：

| 文件 | 未压缩 | 行数 | 列数 | 说明 |
|---|---:|---:|---:|---|
| `reviews.parquet` | 579.58 MB | 2,076,802 | 19 | 评论主表（原始抓取字段） |
| `reviews_tree.parquet` | 562.57 MB | 2,076,802 | 19 | 同量评论，geo 字段为字典编码，**多出 `language` / `language_code`** |
| `review_images.parquet` | 43.91 MB | — | — | 评论图片 URL |

## 2. 字段（reviews.parquet）

`review_id(int64)`, `reviewer_name`, `trip_type`, `publish_time`, `rating`, `title`,
`review_content`, `review_language`, `reviewer_location`, `review_image_count`,
`review_image_urls`, `attraction_name`, `attraction_url`, `continent_name`, `country_name`,
`city_name`, `review_key`, `reviewer_profile_url`, `source_url`

## 3. 字段（reviews_tree.parquet）

`continent`(dict), `country`(dict), `city`(dict), `attraction_name`, `attraction_url`,
`review_id`, `review_key`, `reviewer_name`, `reviewer_profile_url`, `review_date`, `rating`,
`title`, `review_text`, `reviewer_location`, `trip_type`, `published_at`, `language`,
`language_code`, `source_url`

## 4. 已确认的三件事

1. **数据来源是 tripadvisor.cn**（`source_url` 为 `https://www.tripadvisor.cn/Attraction_Review-…`），
   景点名/城市/国家字段是中文，评论语言是「原始语言」而非统一翻译。
   → **原文/自动翻译并未作为两条独立记录存在**（两个 parquet 都只有单一正文字段），
   因此不存在"把同一评论的中英译文当两个样本"的风险；但仍需按 `language` 字段区分语言。
2. 抽样核对：`review_language=英语` 的评论正文确实是英文（非机翻），说明语言标签可用。
3. 规模足够（207 万条），景点粒度可由 `attraction_url`（含 `d<id>`）稳定提取。

## 5. 待办（进入旅游域阶段后执行）

- [ ] 语言分布统计（`language` / `language_code` 两套字段一致性核对）
- [ ] 景点数、每景点评论数、城市/国家分布
- [ ] 空文本/损坏文本比例、重复评论率（按 `review_id`、`review_key`、正文哈希三种口径）
- [ ] 原文与译文关系的最终确认（若 `reviews_tree` 存在与 `reviews` 不同的正文，需逐条比对）
- [ ] 决定是否需要把 1 GB zip 上传到 AutoDL（或用本地处理后的子集）
- [ ] 中英文子集抽取方案，以及**真实 VAD 评价金标准的可行性**（当前无标签，准确性评价必须标"待验证"）
