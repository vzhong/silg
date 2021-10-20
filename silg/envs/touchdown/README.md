# Touchdown Gym Env

To download the data files run in the `touchdown` folder:
```
bash download.sh
```

Then after importing `silg` you should be able to make an environment as follows:
```python
env = gym.make("td_res50_train-v0", cache_size=8, channels=5)
```
where cache size is the size of the LRU cache used to store the features, and channels is the number of (top) components to use in the PCA transformed ResNet-50 features. The observation consists of the following:

- `features`: features
- `x`: the possible (sorted, left to right) coordinates to select. This will always be a tensor of dim `max_actions` padded with `-1`.
- `x_len`: the number of valid actions in `x`. (Will always be in the first `x_len` positions).
- `cur_x`: the current heading. This is undefined for the initial state, so this is selected randomly in the observation returned right after a call to `reset()`.
- `text`: the tokenized text.
- `text_len`: length of the text (without padding).

Everything else about the environment regarding construction and usage should be identical to the other environments.

To test run the environment, run the `gym_touchdown.py` as a script:
```
python gym_touchdown.py --env_id td_res50_train-v0 --cache_size 1 --max_eps 100 --channels 5 --check_max_token
```
If the flag `--check_max_token` is set, we will run the tokenizer on all possible text inputs to compute a value for the `self.max_text`.
The script will also return performance metrics. Running the above script on our machine, we have the following:

```
Max number of tokens: 552

Testing feature loading...
Features size: (47, 128, 5)

constructing cache...
1 GB cache will store 8924 elements. (30% coverage)

===== Graph loaded =====
Number of nodes: 29641
Number of edges: 61319
========================
loading shortest paths file...this might take a while...

finished loading env!

Took 11.494492769241333s to make env.

Testing env with random agent ...
100%|███████████████████████████████████████████████████████████████████████████████████| 100/100 [03:10<00:00,  1.90s/it]
done! average 0.015096811144927062s per step 66.2391541101067 steps per s

cache info: CacheInfo(hits=10559, misses=1967, maxsize=8924, currsize=1967)

used 2405.71484375 MB of RAM
```
