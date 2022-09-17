from utils import helpers
from utils.utils import pload


def get_pretrain(cname, dname, num_clfs):
    name = f"{cname}_{dname}_{num_clfs}.pickle"
    return pload(name, './data/pretrain')

