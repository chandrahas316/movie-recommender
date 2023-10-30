Search Engine for Movies
--------------------------------------------------------------------------------------------------
***Information Retrieval System for a Movie ***

**Problem Statement**:

Building a search engine that will serve a certain domain's
demands is the work at hand. Documents providing data about the
specified domain must be fed to your IR model. The data will then
be processed, and indexes created. The user will then type a query
when this is finished. The top 10 pertinent papers should be
returned as the output.

--------------------------------------------------------------------------------------------------
**How to run the code**
--------------------------------------------------------------------------------------------------


1. Run files in the order: 

              python3 preprocess.py
              python3 tfidf.py
              python3 server.py
2. In your browser go to `http://localhost:3000/`
3. Type your query in the search bar and wait till it returns the relevant documents.

---------------------------------------------------------------------------------------------------
**Dependencies/modules used**
---------------------------------------------------------------------------------------------------
- nltk
- pandas
- pickle
- Numpy
- heapq
- flask
