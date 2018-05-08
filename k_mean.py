import pandas as pd
import sklearn
import json
import sys
import urllib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

def k_mean(k, given_centroids, data_dict):
    
    while(True):
        internal_cluster = {}
        for j in range(k):
            internal_cluster[given_centroids[j]] = []

        for i, t in data_dict.items():
            minimum = sys.maxsize
            identity = 0
            for j in range(k):            
                dist = jaccard_distance(data_dict[given_centroids[j]].split(), t.split())
                if(dist < minimum):
                    minimum = dist
                    identity = given_centroids[j]
            internal_cluster[identity].append(i)
        new_centroids = []
        new_centroids = find_centroid(internal_cluster, data_dict)
        centroid_dist = jaccard_distance(new_centroids, given_centroids)
        if(centroid_dist == 0):
            break
        given_centroids=new_centroids        
    
    return internal_cluster
    
def find_centroid(internal_cluster, data_dict):
    new_centroids = []
    for i, t in internal_cluster.items():
        lists = t
        min_sum = sys.maxsize
        identity = 0
        for j in range(len(lists)):
            sum_distance = 0
            for k in range(len(lists)):
                sum_distance += jaccard_distance(data_dict[lists[j]].split(), data_dict[lists[k]].split())
            if(min_sum > sum_distance):
                min_sum = sum_distance
                identity = lists[j]
        new_centroids.append(identity)
    return new_centroids
                
def jaccard_distance(a, b):
    same = set(a).intersection( set(b) )
    different = set(a).symmetric_difference( set(b) )
    jd = 1- (len(same)/(len(same) + len(different)))
    return jd


if __name__ == "__main__":
    if (len(sys.argv) != 5):
        sys.exit("Not enough arguments, Please enter five arguments")
    else:
        k = int(sys.argv[1])
        tweet_url = sys.argv[3]
        centroid_url = sys.argv[2]
        output_url = sys.argv[4]
#     k = 25
#     tweet_url = "http://www.utdallas.edu/~axn112530/cs6375/unsupervised/Tweets.json"
#     centroid_url = "http://www.utdallas.edu/~axn112530/cs6375/unsupervised/InitialSeeds.txt"
#     output_url = "C:/Users/Rishav/tweets-k-means-output.txt"
    print("Please wait, Code is working!")
    data_dict = {}
    hdr = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',
       'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
       'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
       'Accept-Encoding': 'none',
       'Accept-Language': 'en-US,en;q=0.8',
       'Connection': 'keep-alive'}
    req = urllib.request.Request(tweet_url, headers=hdr)    
    page = urllib.request.urlopen(req)
    line = page.readline()
    while line:
        item = json.loads(line)
        data_dict[item['id']] = item['text']
        line = page.readline()
        
    given_centroids = []
    req = urllib.request.Request(centroid_url, headers=hdr)    
    page = urllib.request.urlopen(req)
    line = page.readline()
    while line:
        given_centroids.append(int(str(line).replace("," ,"").replace("\\n", "").replace("'","").replace("b","")))
        line = page.readline()
    cluster = {}
    cluster = k_mean(k, given_centroids, data_dict)
    sse = 0.0
    for i, t in cluster.items():
        lists = t
        for j in range(len(lists)):
            temp_dist= jaccard_distance(data_dict[lists[j]].split(), data_dict[i].split())
            sse += temp_dist*temp_dist
            
    fo = open(output_url, "w")
    for k, v in cluster.items():
        fo.write(str(k) + '\t' + str(v).replace("[","").replace("]","") + '\n')
    fo.write('SSE Value: ')
    fo.write(str(sse))
    fo.close()

    print("Your wait has paid off, Please check the output file for results!")

