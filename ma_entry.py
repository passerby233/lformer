import os
import time
import argparse
from multiprocessing import Process
import numpy as np
import moxing as mox

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.environ["NCCL_NSOCKS_PERTHREAD"] = "4"
os.environ["NCCL_SOCKET_NTHREADS"] = "2"

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def mp_run(func, args, num_p=7):
    # processing with multiprocessing
    # args: Tuple, args[0] is a index list
    data = args[0]
    seperators = np.linspace(0, len(data), num_p+1).astype(np.int32)  # divide data [num_p] process
    ranges = [(seperators[i], seperators[i+1]) for i in range(len(seperators)-1)]  # data range for each process
    
    process_list = []
    for pid in range(num_p):
        left, right = ranges[pid]
        partial_data = data[left: right]
        sub_args = (pid, partial_data) + args[1:]
        p = Process(target=func, args=sub_args)
        p.start()
        process_list.append(p)
    for p in process_list:
        p.join()

def prepare_extend_args(args, unknown):
    extend_args = [f"--base={args.base}"]
    extend_args.extend([f"--world_size={args.world_size}"])
    for ukn in unknown:
        extend_args.append(ukn)
    
    if len(args.train_url):
        extend_args.extend([f"--train_url={args.train_url}"])
    if len(args.data_url):
        extend_args.extend([f"--data_url={args.data_url}"])
    
    print("MA train entry extended args", extend_args)
    return extend_args     

def release(filepath, target_dir):
    # release a compressed file
    print(f"Releasing {filepath} to {target_dir}")
    start_t = time.time()
    if filepath.split('.')[-1] == 'tar':
        os.system(f"tar -xvf {filepath} -C {target_dir} > /dev/null")
    else:
        os.system(f"unzip -q -d {target_dir} {filepath}")
    print(f"Releasing {filepath} costs {time.time()-start_t} s.") 

def download(source_path, target_path):
    # download with mox module
    start_t = time.time()
    mox.file.copy_parallel(source_path, target_path)
    print(f"Downloading {target_path} costs {time.time()-start_t} s.") 

def download_and_release(source_dir, target_dir, filename):
    source_path = os.path.join(source_dir, filename)
    target_path = os.path.join(target_dir, filename)
    download(source_path, target_path)
    release(target_path, target_dir)

def prepare_cc3m(source_dir, target_dir):
    def untar_cc3m(pid, data, source_dir, target_dir):
        for idx in data:
            filepath = os.path.join(source_dir, f"{idx}.tar")
            release(filepath, target_dir)
    package_indexs = [0, 3, 6, 9, 12, 15, 18]
    args = (package_indexs, source_dir, target_dir)
    mp_run(untar_cc3m, args, num_p=7)

    source_s3 = "s3://bucket-3947/lijiacheng/datasets/cc3m/"
    download_and_release(source_s3, target_dir, 'val2017.zip')
    download_and_release(source_s3, target_dir, '21.tar')

def enter_node_barrier(args):
    s3_root = "s3://bucket-3947/lijiacheng/sync/"
    sync_dir = os.path.join(s3_root, 'node_ready_flag', os.environ['MA_VJ_NAME'])
    if args.rank == 0:
        assert not mox.file.exists(sync_dir)
        mox.file.make_dirs(sync_dir)
    else:
        while not mox.file.exists(sync_dir):
            print(f"wait for directory {sync_dir} to exist")
            time.sleep(10)
    with mox.file.File(os.path.join(sync_dir, str(args.rank)), 'w') as f:
        f.write(f"args.rank")
    ready_nodes = mox.file.list_directory(sync_dir)
    while len(ready_nodes) != args.world_size:
        print(f"[{len(ready_nodes)}/{args.world_size}] nodes are ready, {ready_nodes} is waiting for others." )
        time.sleep(10)
        ready_nodes = mox.file.list_directory(sync_dir)
    if args.rank == 0:
        time.sleep(20)
        mox.file.remove(sync_dir, recursive=True)

def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--main', type=str, default='main.py', help='lauch script')
    parser.add_argument('--train_url', type=str, default='', help='download path')
    parser.add_argument('--data_url', type=str, default='', help='upload path')
    parser.add_argument('--init_method', default='tcp://127.0.0.1:6666', help='tcp_port')  
    parser.add_argument('--world_size', type=int, default=1, help='total number of nodes')                  
    parser.add_argument('--rank', type=int, default=0, help='the index of node')
    parser.add_argument("--base", type=str, default="", help="config file")

    # Get args
    args, unkown = parser.parse_known_args()
    print("args", args, 'unkown', unkown)
    extend_args = prepare_extend_args(args, unkown)

    # Init for ModerArst Environment
    addr, port = args.init_method.split('//')[-1].split(':')
    os.environ['MASTER_ADDR'] = addr
    os.environ['MASTER_PORT'] = port
    os.environ['GROUP_RANK'] = str(args.rank)
    print(addr, port, os.environ['GROUP_RANK'])

    # Prepare data
    if 'cc3m' in args.base:
        source_dir = "/home/ma-user/work/zhanzongyuan/cc3m/"
        target_dir = "/cache/cc3m/"
        if not os.path.exists(target_dir):
            os.makedirs(target_dir, exist_ok=True)
        prepare_cc3m(source_dir, target_dir)
        print(os.listdir(target_dir))
    elif 'celeba' in args.base:
        source_dir = "s3://bucket-3947/lijiacheng/datasets/celeba/"
        target_dir = "/cache/"
        download_and_release(source_dir, target_dir, "CelebAMask-HQ.zip")
        print(os.listdir(target_dir))
    elif 'coco' in args.base:
        source_dir = "s3://bucket-3947/lijiacheng/datasets/coco/"
        target_dir = '/cache/coco/'    
        for zipfile in ['annotations_trainval2017.zip', 'train2017.zip', 'val2017.zip']:
            download_and_release(source_dir, target_dir, zipfile)
        print(os.listdir(target_dir))
    elif 'imagenet' in args.base:
        source_dir = "s3://bucket-3947/wenyijiang/mycode/sunshikun/Data/ImageNet/"
        target_dir = '/cache/imagenet/'    
        for zipfile in ['val.tar', 'train.tar']:
            download_and_release(source_dir, target_dir, zipfile)
        print(os.listdir(target_dir))
    elif 'laion' in args.base:
            source_dir = 's3://bucket-3690/zhanzongyuan/Datasets/laion/mm_en_filename_caption.lmdb'
            target_dir = '/cache/mm_en_filename_caption.lmdb/'
            download(source_dir, target_dir)
            print(os.listdir(target_dir))

    # wait for all node sync
    if 'cub' not in args.base:
        enter_node_barrier(args)
    #download("s3://bucket-3947/lijiacheng/pretrained", "/cache/pretrained")
    
    # Run the script
    cmd = f"python {args.main}"
    cmd = cmd + ' ' + ' '.join(extend_args)
    print("====> EXEC CMD", cmd)
    os.system(cmd)

if __name__ == '__main__':
    main()  