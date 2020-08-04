import random
import numpy as np
import pandas as pd
import operator

class UserCF():
    def __init__(self,datafile):
        self.datafile = datafile
        self.sim_n_movies = 20
        self.rec_n_movies = 10

        self.trainset = {}
        self.testset = {}

        self.user_sim_matrix = {}


    def load_file(self):
        with open(self.datafile,'r') as f:
            for i,line in enumerate(f):
                if i == 0:
                    continue
                yield line.strip('\r\n')

    def get_data(self,p):
        train_len,test_len = 0,0
        for line in self.load_file():
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

    def calculate_user_sim(self):
        movies_user = {}
        for user,movies in self.trainset.items():
            for movie in movies:
                if movie not in movies_user:
                    movies_user[movie] = set()
                movies_user[movie].add(user)
        print('Movies_user table done!')
        self.movie_count = len(movies_user)
        for movie,users in movies_user.items():
            for user in users:
                for other_user in users:
                    if user == other_user:
                        continue
                    self.user_sim_matrix.setdefault(user,{})
                    self.user_sim_matrix[user].setdefault(other_user,0)
                    self.user_sim_matrix[user][other_user] += 1/np.log(1+len(users))
        print('User_Sim_Matrix done!')

        for user,related_users in self.user_sim_matrix.items():
            for u,count in related_users.items():
                self.user_sim_matrix[user][u] = count/np.sqrt(len(self.trainset[user])*len(self.trainset[u]))

        print('Calculate user similarity matrix done!')

    def recommand(self,user_id,topK_users=10,topK_movies=15):
        rank = {}
        seen = self.trainset[user_id]
        for related_user,score in sorted(self.user_sim_matrix[user_id].items(),
                                         key=operator.itemgetter(1),reverse=True)[:topK_users]:
            for movie in self.trainset[related_user]:
                if movie in seen:
                    continue
                rank.setdefault(movie,0)
                rank[movie] += score
        return sorted(rank.items(),key=operator.itemgetter(1),reverse=True)[:topK_movies]

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
            rec_count += 15
            test_count += len(test_movies)
        precision = hit/(1.0*rec_count)
        recall = hit/(1.0*test_count)
        coverage = len(all) / (1.0 * self.movie_count)
        print('precisioin=%.4f\trecall=%.4f\tcoverage=%.4f' % (precision, recall, coverage))


if __name__=='__main__':
    userCF = UserCF('ratings.dat')
    userCF.get_data(0.8)
    userCF.calculate_user_sim()
    userCF.evaluate()
