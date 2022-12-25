import numpy as np
import random
import ges
np.random.seed(seed=2022)

def map_randomly(dom_f, img_f):
    f = dict((x, random.choice(img_f)) for x in dom_f)
    # ensure that f is not a constant function
    if len(set(f.values())) == 1:
        f = map_randomly(dom_f, img_f)
    assert len(set(f.values())) != 1
    return f

def generate_sequence(dom_size: int, size: int):
    # generate X from multinomial disctibution
    p_nums = [np.random.random() for _ in range(dom_size)]
    p_vals = [v / sum(p_nums) for v in p_nums]
    X = np.random.choice(a=range(dom_size), p=p_vals, size=size)
    return X

def test_model5(args):
    X = generate_sequence(args.x_dom, args.sample_size)
    f1 = map_randomly(range(args.x_dom), range(args.y_dom))
    f2 = map_randomly(range(args.x_dom), range(args.z_dom))
    E_Y = generate_sequence(args.y_dom, args.sample_size)
    E_Z = generate_sequence(args.z_dom, args.sample_size)
    Y = [(f1[x] + e_y) % args.y_dom for x, e_y in zip(X, E_Y)]
    Z = [(f2[x] + e_z) % args.z_dom for x, e_z in zip(X, E_Z)]

    data = np.stack([X, Y, Z], axis=-1)
    return data




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="systhetic experiment")
    parser.add_argument("--x_dom", type=int, default=5, help="x domain size")
    parser.add_argument("--y_dom", type=int, default=6, help="y domain size")
    parser.add_argument("--z_dom", type=int, default=6, help="z domain size")
    parser.add_argument("--sample_size", type=int, default=1000, help="sample size")
    args = parser.parse_args()

    data = test_model5(args)
    estimate, score = ges.fit_nml(data, A0=np.array([[0, 1, 1], [0, 0, 0], [0, 0, 0]]))

    print(estimate, score)
