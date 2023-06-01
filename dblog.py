import sqlite3 
import numpy as np
import json
class DbLog:
    def __init__(self,fname):
        super().__init__()
        print(f"opening {fname}")
        self.con = sqlite3.connect(fname)
        self.con.cursor().execute('''CREATE TABLE IF NOT EXISTS log (tag text, ind real,val BLOB,t TIMESTAMP DEFAULT CURRENT_TIMESTAMP,UNIQUE(tag, ind) ON CONFLICT REPLACE)''')
        self.con.commit()
    def log(self,tag,val,ind):
        self.con.cursor().execute(f"INSERT OR REPLACE INTO log(tag,ind,val) VALUES (?,?,?)",(tag,ind,json.dumps(val)))
        self.con.commit()
    def read(self,tag):
        return list(self.con.cursor().execute(f"select ind,val from log where tag = '{tag}'"))
    def keys(self):
        return list(map(lambda x: x[0],list(self.con.cursor().execute(f"select distinct tag from log"))))

    def __del__(self):
        self.close()

    def close(self):
        self.con.close()
if __name__ == '__main__':
    logger = DbLog('test.db')
    logger.log('test',5,20)
    logger.log('test',6,21)
    logger.log('test',5,25)
    logger.log('test',5,25)
    print(logger.read('test'))
    dat = logger.read('test')
    print(np.array(dat).astype(np.float32).astype(int))
    
