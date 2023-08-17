from subprocess import PIPE, Popen
import joblib


n_parallel = 1
n_files = 100


# 各入力ごとのスコアを計算する
def calc_score_each(seed: int):
    p = Popen(f"cargo run --release --bin tester ../a.out < in/{seed:04}.txt > out.txt", shell=True, stdout=PIPE, stderr=PIPE)
    stdout, stderr = p.communicate()
    score_part = stderr.decode().split("\n")[-6]
    print(seed, score_part)
    return float(score_part.split()[-1])


# 平均スコアを計算する
def calc_scores():
    scores = joblib.Parallel(n_jobs=n_parallel)(
        joblib.delayed(calc_score_each)(i) for i in range(n_files)
    )
    return scores


if __name__ == "__main__":
    Popen("g++ -O2 -std=c++17 ../main.cpp -o ../a.out", shell=True).wait()
    scores = calc_scores()
    print(sum(scores) / len(scores))