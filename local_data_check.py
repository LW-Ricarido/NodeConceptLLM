from datasets import load_from_disk
from tqdm import tqdm
ds = load_from_disk('../../LLM4Graph_tracked/datasets_local/all_downstream_dataset_32')
for i in tqdm(range(len(ds))):
    embedding_length = len(ds[i]['unaligned_input_embeds'])
    mask_length = ds[i]['question'].count('<|embedding_mask|>')

    if embedding_length != mask_length:
        print('==================wtf')
        import ipdb; ipdb.set_trace()