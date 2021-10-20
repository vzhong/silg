import os
import pprint
import submitit
import hashlib
import argparse
import itertools
import run_exp
from expman import Experiment, JSONLogger


parser = argparse.ArgumentParser()
parser.add_argument('--seeds', default=4, type=int)
parser.add_argument('--time', default=60*24*3, help='timeout in minutes', type=int)
parser.add_argument('--envs', nargs='+', default=['rtfm'], help='environments to run')
parser.add_argument('--partition', help='slurm partition')
parser.add_argument('--savedir', default='checkpoint', help='where to save jobs')
parser.add_argument('--dry', action='store_true', help='dont actually launchjobs')
parser.add_argument('--constraint', help='optional slurm constraints')
parser.add_argument('--force_launch', action='store_true', help='force launch jobs that are still running (or pinged very recently)')
parser.add_argument('--local', action='store_true', help='run on local machine')
parser.add_argument('--wandb', action='store_true', help='log to wandb')


def get_opts(**opts):
    keys = sorted(list(opts.keys()))
    flat = [opts[k] for k in keys]
    prod = itertools.product(*flat)
    out = []
    for p in prod:
        assignment = dict(zip(keys, p))
        concat = []
        for k in sorted(list(assignment.keys())):
            concat.append('{}-{}'.format(k, assignment[k]))
        concat = ';'.join(concat)
        prefix = hashlib.md5(concat.encode('utf-8')).hexdigest()
        out.append((prefix, assignment))
    return out


if __name__ == '__main__':
    args = parser.parse_args()
    os.environ['OMP_NUM_THREADS'] = '1'

    if not args.local:
        executor = submitit.SlurmExecutor(folder=os.path.join(args.savedir, 'slurm'), max_num_timeout=5)

    slurm_kwargs = {
        'job_name': 'silg',
        'nodes': 1,
        'gpus_per_node': 1,
        'ntasks_per_node': 1,
        'cpus_per_task': 1,
        'partition': args.partition,
        'mem': '64G',
        'time': args.time,
        'constraint': args.constraint,
    }

    pprint.pprint(slurm_kwargs)

    opts_list = {}
    opts_list['rtfm'] = get_opts(
        model=['multi'],
        entropy_cost=[0.05, 0.005],
        stateful=[False],
        use_local_conv=[False],
        field_attn=[False],
        env=['silg:rtfm_train_s1-v0'],
        val_env=['silg:rtfm_test_s1-v0'],
    )
    opts_list['msgr'] = get_opts(
        model=['multi'],
        entropy_cost=[0.05, 0.005],
        stateful=[False],
        use_local_conv=[False],
        field_attn=[False],
        env=['silg:msgr_train-all_s1-v0'],
        val_env=['silg:msgr_dev_s1-v0'],
    )
    opts_list['nethack_old'] = get_opts(
        model=['multi'],
        entropy_cost=[0.05, 0.005],
        stateful=[False],
        use_local_conv=[False],
        field_attn=[False],
        num_actors=[8],
        batch_size=[8],
        unroll_length=[64],
        env=['silg:nethack_train-v0'],
        val_env=['silg:nethack_dev-v0'],
    )
    opts_list['nethack_new'] = get_opts(
        model=['multi'],
        entropy_cost=[0.05, 0.005],
        stateful=[False],
        use_local_conv=[False],
        field_attn=[False],
        num_actors=[8],
        batch_size=[8],
        unroll_length=[64],
        env=['silg:nethack_train_new-v0'],
        val_env=['silg:nethack_dev_new-v0'],
    )
    opts_list['alfworld2'] = get_opts(
        model=['multi_rank'],
        entropy_cost=[0.001, 0.005],
        stateful=[False],
        use_local_conv=[False],
        field_attn=[False],
        batch_size=[10],
        eval_eps=[140],
        env=['silg:alfworld_train-v0'],
        val_env=['silg:alfworld_test_id-v0'],
    )
    opts_list['touchdown'] = get_opts(
        model=['multi_nav_emb'],
        entropy_cost=[0.05, 0.005],
        learning_rate=[0.0005],
        stateful=[False],
        use_local_conv=[False],
        field_attn=[False],
        demb=[30],
        drnn=[100],
        drep=[200],
        num_actors=[8],
        batch_size=[3],
        unroll_length=[64],
        num_film=[3],
        env=['silg:td_segs_train-v0'],
        val_env=['silg:td_segs_dev-v0'],
    )

    n = 0
    for env in args.envs:
        opts = opts_list[env]
        for prefix, opt in opts:
            flags = run_exp.exp_utils.get_parser().parse_args([])
            flags.wandb = args.wandb

            for k, v in opt.items():
                assert hasattr(flags, k), 'Invalid spec {}={}'.format(k, v)
                setattr(flags, k, v)

            flags.savedir = args.savedir
            if not args.local:
                slurm_kwargs['cpus_per_task'] = flags.num_actors + 1
                executor.update_parameters(**slurm_kwargs)

            for seed in range(args.seeds):
                flags.xpid = '{}-{}-{}-{}'.format(env, flags.model, prefix, seed)
                if not args.local:
                    executor.folder = os.path.join(flags.savedir, flags.xpid, 'slurm')

                # check if job already exists
                exp = Experiment.from_namespace(flags, name_field='xpid', logdir_field='savedir')
                if exp.exists():
                    try:
                        exp.load()
                    except Exception as e:
                        print('could not load {}'.format(exp.explog))
                        raise e
                    if exp.last_written_time is not None:
                        print('Found existing experiment {}'.format(exp.explog))
                        if exp.time_since_last_written().seconds > 2000:
                            logs = exp.load_logs(JSONLogger(), error='ignore')
                            m = max([r['frames'] for r in logs]) if logs else 0
                            if logs and m >= 0.9 * flags.total_frames:
                                print('Job is done at {} frames'.format(m))
                                continue
                            else:
                                print('It has been {}s since this job has responded at {} frames. Relaunching'.format(exp.time_since_last_written().seconds, m))
                        elif not args.force_launch:
                            print('This job is still ongoing')
                            continue
                else:
                    exp.save()

                print('Launching job {}: {}'.format(n+1, flags))

                if not args.dry:
                    job = run_exp.Train()
                    if args.local:
                        job(exp.explog)
                    else:
                        job.launch_slurm(exp.explog, executor=executor)
                n += 1
    print('launched {} jobs{}'.format(n, ' (but not really because --dry)' if args.dry else ''))
