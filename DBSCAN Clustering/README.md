Project Overview

This project computes pairwise Wasserstein distance matrices across languages for three Wikipedia-derived corpora:

Sports

Ideologies

Historical Objects

The distances are based on distributions of year mentions extracted from tagged datasets. These matrices allow us to quantitatively compare historical emphases across languages and serve as the foundational input for DBSCAN clustering, avoiding the need to recompute distances repeatedly.
In addition to CSV outputs, the code also generates PNG visualizations for quick inspection.

Key Objectives

Extract valid years from multilingual historical datasets (filtering: 32 ≤ year ≤ 2025).

Build language–year frequency distributions for each language.

Compute Wasserstein distances (Earth Mover’s Distance) between every pair of languages.

Generate both CSV matrices and PNG heatmaps for later use in clustering and visualization.

Provide distance dictionaries so that DBSCAN clustering can be performed directly without recomputation.

File Structure

Input datasets

filename_dates_packed_history_of_sports_tagged.csv

filename_dates_packed_history_of_ideologies_tagged.csv

filename_dates_packed_historical_objects_tagged.csv

Word count CSVs

filename_wordcounts_sports.csv

filename_wordcounts_ideologies.csv

filename_wordcounts_objects.csv

Outputs

CSV distance matrices (e.g., wasserstein_sports_top25.csv)

PNG heatmaps (e.g, wasserstein_sports_top25.png)

How It Works
1. Language Ranking

Wordcount files rank languages by total token counts.

Three scopes:

top10 = top 10% of languages

top25 = top 25% of languages

top50 = top 50% of languages

2. Year Extraction

Regex extraction of years (\b\d{3,4}\b).

Valid years: 32–2025 inclusive.

3. Pivot Table Construction

Builds a language × year pivot table.

Rows = languages, columns = years, values = mention counts.

4. Wasserstein Distance

For each language pair, constructs weighted year distributions.

Computes Wasserstein-1 distance.

Handles empty distributions (0 if both empty, ∞ if only one non-empty).

5. Outputs

For each corpus (sports, ideologies, objects) and scope (top10, top25, top50), the script produces:

CSV Distance Matrices

Example: wasserstein_ideologies_top10.csv

Used as input dictionaries for DBSCAN clustering.

Clustering Visualizations (DBSCAN)

Example: dbscan_ideologies_top10_original.png

Example: dbscan_ideologies_top10_filtered.png

The original PNG shows clustering results on the full dataset.

The filtered PNG shows clustering results after removing outliers, giving a clearer structure.

Visualization Style

The clustering visualization is presented as a 2D scatter plot after dimensionality reduction.

Each point = one language corpus.

Labels = language filenames (English.txt, Chinese.txt, etc.).

Colors = DBSCAN cluster assignments.

Outliers appear as singletons in the original plot and are removed in the filtered version.
Both CSV and PNG files are saved for each group–scope combination.
