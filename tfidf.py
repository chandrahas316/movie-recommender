import time
import numpy as np
import pickle

def load_pickle(doc):
    f = open(doc,'rb')
    df = pickle.load(f)
    f.close()
    return df

def tf_idf(processed_data, inverted_index, length):
    print("Time taken to create tf-idf score")
    start_time = time.time()
    no_of_doc = 34886

    # Loading term frequencies
    df = load_pickle(processed_data)

    d_df = df.to_dict()
    d_df =d_df[length]
    

    # Average length of all documents
    avg_length= df[length].mean() 
    

    ii_df = load_pickle(inverted_index)# indexing list

    ii_df= ii_df.to_dict()
    ii_df=ii_df['PostingList'] 

    
    tf_idf_dict={}

    # Calculating tf-idf
    for doc in range(0,no_of_doc):
        doc_dict={}
        for key,value in df['Frequency'][doc].items():
            if key=='nan' or key=='null':
             continue
            tf = (value/d_df[doc]) 
            idf = np.log(no_of_doc/(ii_df[key]))
            doc_dict[key] = tf * idf
        tf_idf_dict[doc]=doc_dict

    print("--- %s seconds ---" % (time.time() - start_time))
    return tf_idf_dict  

def run():
    tf_idf_dict = tf_idf("processed_data.obj", "inverted_index.obj", "Length")
    filehandler = open("tf-idf.obj","wb")
    pickle.dump(tf_idf_dict,filehandler)
    filehandler.close()

    tf_idf_title_dict = tf_idf("processed_data_title.obj", "inverted_index_title.obj", "TitleLength")
    filehandler = open("tf-idf_title.obj","wb")
    pickle.dump(tf_idf_title_dict,filehandler)
    filehandler.close()

run()



