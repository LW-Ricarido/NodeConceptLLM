from google.cloud import bigquery
import pandas as pd
import os 
root_path = 'your_local_root_path/ogbl_citation2/mapping'


query = """
SELECT 
    json_value(work,"$.title") as title,
    json_extract(work, "$.abstract_inverted_index") as abstract,
    json_value(work, "$.ids.mag") as mag_id
FROM openalex-demo-458904.openalex.works
WHERE JSON_EXTRACT_SCALAR(work, '$.ids.mag') in ('2905649489','2063597572')
ORDER by mag_id
"""

def get_title_abstract_data(local_rank,total_works):
    batch_query_size = 20
    nodeid2magid = pd.read_csv(os.path.join(root_path,'nodeidx2paperid.csv.gz'),compression='gzip')
    ds_length = len(nodeid2magid)
    local_process_length = ds_length // total_works
    bos = local_rank * local_process_length
    eos = min(local_rank *local_process_length, ds_length)
    for i in range(bos, eos,batch_query_size):
        magid_2_nodeid = {}
        query_ids = ''
        for j in range(i,min(eos, (i+1) * batch_query_size)):
            local_id = nodeid2magid.iloc[j]
            magid_2_nodeid[local_id['paper id']] = local_id['node idx']
        
    ###