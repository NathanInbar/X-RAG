import json, ijson
from xrag.utils.models import SPOTriple, Chunk
from xrag.utils import normalize_from_name
from xrag.utils import mg_driver

async def upsert_from_preprocessed(input_json, descriptions_json):
    with open(input_json, "rb") as in_file, open(descriptions_json, "r") as descriptions_file:
        # load description map
        data = json.load(descriptions_file)
        description_map = {}
        for obj in data:
            description_map.update(obj)
        del data

        #upsert all triples
        for itm in ijson.items(in_file, "item"):
            print(f"ingesting {itm['source']} ({len(itm['chunks'])} chunks) (id:{itm['id']})")

            for i,chunk in enumerate(itm['chunks']):
                chunk:Chunk = chunk
                for triple in chunk['triples']:
                    snorm , onorm = normalize_from_name(triple['s']),normalize_from_name(triple['o'])
                    pnorm = snorm + "__" + normalize_from_name(triple['p']) + "__" + onorm
                    norm_triple:SPOTriple = { 's':snorm, 'p':pnorm, 'o':onorm }
                    try:
                        triple_descriptions = (description_map[norm_triple['s']], description_map[norm_triple['p']], description_map[norm_triple['o']])
                    except KeyError:
                        print(f"couldnt get description for a triple element! triple: {norm_triple}")
                        continue    
                    await mg_driver.merge_triple(triple, norm_triple, triple_descriptions, source_doc_id=itm['id'], source_chunk_id=chunk['id'])
                print(f"{".." if i%20!=0 else "\n.."}{i}",end="")