# RecSys Bundle
Bundle of recommender systems implemented (mostly) from scratch. The dataset I used is 
[MovieLens Latest Small](https://grouplens.org/datasets/movielens/).

## Installation
For convenience I provide an environment file, which can be used to create a [Conda](https://www.anaconda.com/docs/getting-started/miniconda/install) environment:

```bash
conda env create -f environment.yml
conda activate rec-sys
```

## Recommendation Problem
TODO

### Rating Prediction
TODO

### Ranking
TODO

## Models

### Collaborative Filtering
Collaborative filtering models produce the recommendations by learning the user-item interactions matrix. The main idea
is that similar users will probably share preferences.

#### Matrix Factorization
Matrix factorization is a technique which aims to decompose the user-item interactions matrix into two smaller matrices
called embeddings, $P$ users and $Q$ items. The goal is to have that $R \approx P \cdot Q$, so that recommendations can
then be predicted as:

$$\hat{r}_{ui} = \mu + p_u \cdot q_i^T$$

where $\mu$ is the global average rating.

##### Stochastic Gradient Descent
By minimizing the Mean Square Error (MSE) through stochastic gradient descent (SGD) it is possible to obtain an
approximation of $P$ and $Q$. This implementation also features users' and items' biases, which should allow to better
learn the train dataset's patterns.

Update rules for each epoch:
```math
\begin{alignat*}{3}
  & P_u \, &=& \, P_u + 2 \eta \left( \epsilon_{ui} \cdot Q_i - \lambda \cdot P_u \right) \\
  & Q_i \, &=& \, Q_i + 2 \eta \left( \epsilon_{ui} \cdot P_u - \lambda \cdot Q_i \right) \\
  & b_u \, &=& \, b_u + \eta \left( \epsilon_{ui} - \lambda \cdot b_u \right) \\
  & b_i \, &=& \, b_i + \eta \left( \epsilon_{ui} - \lambda \cdot b_i \right)
\end{alignat*}
```

where $\eta$ is the learning rate, $\epsilon_{ui}$ stands for the prediction errors for the users and items involved
in the epoch, and $\lambda$ is the $L_2$ regularization term which helps to prevent overfitting.

The prediction with biases is computed as:

$$\hat{r}_{ui} = b_u + b_i + \mu + p_u \cdot q_i^T$$

##### Alternating Least Squares
Alternating Least Squares (ALS) is a matrix factorization algorithm which fixes one of the two embeddings in order to
optimize the other. By doing so, the optimization step gets simplified to solving a quadratic problem:

```math
\begin{alignat*}{3}
  & P_u \, &=& \, \left( \sum_{r_{ui} \in r_{u*}} Q_i \cdot Q_i^T + \lambda I_k \right)^{-1} \cdot \sum_{r_{ui} \in r_{u*}} r_{ui} \cdot Q_i \\
  & Q_i \, &=& \, \left( \sum_{r_{ui} \in r_{*i}} P_u \cdot P_u^T + \lambda I_k \right)^{-1} \cdot \sum_{r_{ui} \in r_{*i}} r_{ui} \cdot P_u 
\end{alignat*}
```

where $r_{u*}$ are the ratings provided by user $u$, $r_{*i}$ are the ratings assigned to item $i$, $\lambda$ is the $L_2$ 
regularization term which helps to prevent overfitting, and $I$ is the identity matrix.

##### Alternating Least Squares Map Reduce
The main advantage of ALS is that it is incredibly easy to parallelize, as each user/item embedding is computed independentely
of the others. This is a custom Map Reduce implementation using PySpark, which treats the embeddings as Resilient Distributed Datasets
(RDD). In this manner, the workload can be distributed across a cluster of worker machines, and thus expedited.

#### Neighborhood-based
Neighborhood-based models learn user preferences through similarity measurements. The two most popular similarity metrics are
Adjusted Cosine Similarity and Pearson Correlation. The ratings are then predicted using the following formula in the case of
User-based neighborhood:

```math
\hat{r}_{ui} = \mu_u + \frac{\sum_{v \in N_i^k \left( u \right)} \text{sim} \left( u,v \right) \cdot \left( r_{vi} - \mu_v \right)}{\sum_{v \in N_i^k \left( u \right)} \text{sim} \left( u,v \right)}
```

The prediction formula for Item-based neighborhood is analogous.

### Non-Personalized
Non-personalized recommender systems are useful for dealing with the cold-start problem, namely newly-registered users lack the required
number of ratings necessary for the models to learn their preferences.

#### Most Popular
Most popular recommendation is quite simple, as it consists in ranking the items according to the number of ratings they possess.

#### Highest Rated
Ranking according to the highest rated items is not as straightforward, as an item $i$ with only one 5/5 rating should be ranked
lower than another item $j$ with 100 ratings which average to 4.5/5. So the Bayesian average of the ratings is considered instead
of the means in absolute terms:

```math
\frac{\mu_i \cdot n + c \cdot \mu}{n + c}
```

where $\mu_i$ is the average rating for item $i$, $n$ is the total number of ratings for item $i$, and $c$ is the confidence constant,
defined as the first quartile of the rating counts' distribution.

The effect here is that an item with a very low amount of ratings will obtain a Bayesian average which is close to the global
average.

### Content-based Filtering
Content-based filtering techniques focus on building user profiles which are then utilized to compute the most similar new items
to recommend.

The first step is to assign to each item a vectorized representation of its features, in this case I used an unordered combination of
the genres and user-supplied tags. I fed this data to a TF-IDF vectorizer, whose output I further processed using Truncated SVD so that
the final vectors are neither too sparse nor binarized, which would make it harder to obtain averaged vectors when creating the users' profiles.

The users' profiles are built using the ratings' histories of the users, so that by selecting the latest movies which the user has rated,
both positively and negatively, I create a profile which is a weighted average of the positive items' vectors minus the negative items'
vectors.

Then the recommendations are obtained through a K-Nearest-Neighbors model using cosine distance to measure similarity between the users'
profiles and the movies' vectors.

## Results
TODO
