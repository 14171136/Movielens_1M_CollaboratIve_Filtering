import random
import numpy as np
from tqdm import tqdm
import operator

class ItemCF():
    def __init__(self,datafile):
        self.datafile = datafile
        self.sim_n_movies = 20
        self.rec_n_movies = 10

        self.trainset = {}
        self.testset = {}

        self.item_popular = {}
        self.item_sim_matrix = {}


    def load_file(self):
        with open(self.datafile,'r') as f:
            for i,line in enumerate(f):
                if i == 0:
                    continue
                yield line.strip('\r\n')

    def get_data(self,p):
        train_len,test_len = 0,0
        for i,line in enumerate(self.load_file()):
            random.seed(i)
            user,movie,rating,timestamp = line.split('::')
            if random.random() < p:
                self.trainset.setdefault(user,{})
                self.trainset[user][movie] = rating
                train_len += 1
            else:
                self.testset.setdefault(user, {})
                self.testset[user][movie] = rating
                test_len += 1
        print('TrainSet length {0}'.format(train_len))
        print('TestSet length {0}'.format(test_len))

    def calculate_item_sim(self):
        for user,movies in self.trainset.items():
            for movie in movies:
                self.item_popular.setdefault(movie,0)
                self.item_popular[movie] += 1

        self.movie_counts = len(self.item_popular)

        #self.movie_count = len(movies_user)
        for user,movies in self.trainset.items():
            for movie in movies:
                for other_movie in movies:
                    if movie == other_movie:
                        continue
                    self.item_sim_matrix.setdefault(movie,{})
                    self.item_sim_matrix[movie].setdefault(other_movie,0)
                    self.item_sim_matrix[movie][other_movie] += 1
        print('Item_Sim_Matrix done!')

        for movie,related_movies in self.item_sim_matrix.items():
            for m,count in related_movies.items():
                if self.item_sim_matrix[movie] == 0 or self.item_sim_matrix[m] == 0:
                    self.item_sim_matrix[movie][m] = 0
                else:
                    self.item_sim_matrix[movie][m] = count/np.sqrt(self.item_popular[movie] * self.item_popular[m])
        #ItemCF-Norm
        # for movie in tqdm(self.item_sim_matrix.keys()):
        #     _max = max(self.item_sim_matrix[movie].values())
        #     for m in self.item_sim_matrix[movie]:
        #         self.item_sim_matrix[movie][m] /= _max

        print('Calculate item similarity matrix done!')

    def recommand(self,user_id,topK_items=15,topK_users=20):
        rank = {}
        seen = self.trainset[user_id]
        for movie,rating in seen.items():
            for related_movie,score in sorted(self.item_sim_matrix[movie].items(),
                                              key=operator.itemgetter(1),reverse=True)[:topK_items]:
                if related_movie in seen:
                    continue
                rank.setdefault(related_movie,0)
                rank[related_movie] += score*float(rating)
        return sorted(rank.items(),key=operator.itemgetter(1),reverse=True)[:topK_users]



    def evaluate(self):
        print('Evaluation Start...')
        hit = 0
        rec_count,test_count = 0,0
        all = set()
        for i,user in enumerate(self.trainset):
            test_movies = self.testset.get(user,{})
            rec_movies = self.recommand(user)
            for movie,rating in rec_movies:
                if movie in test_movies:
                    hit += 1
                all.add(movie)
            rec_count += 20
            test_count += len(test_movies)
        precision = hit/(1.0*rec_count)
        recall = hit/(1.0*test_count)
        coverage = len(all) / (1.0 * self.movie_counts)
        print('precisioin=%.4f\trecall=%.4f\tcoverage=%.4f' % (precision, recall, coverage))


if __name__=='__main__':
    itemCF = ItemCF('ratings.dat')
    itemCF.get_data(0.8)
    itemCF.calculate_item_sim()
    itemCF.evaluate()
