/* #region header */

#pragma GCC optimize("Ofast")
#include <bits/stdc++.h>
using namespace std;
// types
using ll = long long;
using ull = unsigned long long;
using ld = long double;
typedef pair<ll, ll> Pl;
typedef vector<ll> vl;
typedef vector<int> vi;
typedef vector<char> vc;
template <typename T> using mat = vector<vector<T>>;
typedef vector<vector<int>> vvi;
typedef vector<vector<long long>> vvl;
typedef vector<vector<char>> vvc;
// abreviations
#define all(x) (x).begin(), (x).end()
#define rall(x) (x).rbegin(), (x).rend()
#define rep_(i , a_, b_, a, b, ...) for (ll i = (a); i < b; i++)
#define rep(i, ...) rep_(i, __VA_ARGS__, __VA_ARGS__, 0, __VA_ARGS__)
#define rrep_(i, a_, b_, a, b, ...) for (ll i = (b - 1); i >= a; i--)
#define rrep(i, ...) rrep_(i, __VA_ARGS__, __VA_ARGS__, 0, __VA_ARGS__)
#define srep(i, a, b, c) for (ll i = (a); i < b; i += c)
#define SZ(x) ((int)(x).size())
#define pb(x) push_back(x)
#define eb(x) emplace_back(x)
#define mp make_pair
//入出力
#define print(x) cout << x << endl
template <class T> ostream &operator<<(ostream &os, const vector<T> &v) {
    for (auto &e : v)
        cout << e << " ";
    cout << endl;
    return os;
}
void scan(int &a) {
    cin >> a;
}
void scan(long long &a) {
    cin >> a;
}
void scan(char &a) {
    cin >> a;
}
void scan(double &a) {
    cin >> a;
}
void scan(string &a) {
    cin >> a;
}
template <class T> void scan(vector<T> &a) {
    for (auto &i : a)
        scan(i);
}
#define vsum(x) accumulate(all(x), 0LL)
#define vmax(a) *max_element(all(a))
#define vmin(a) *min_element(all(a))
#define lb(c, x) distance((c).begin(), lower_bound(all(c), (x)))
#define ub(c, x) distance((c).begin(), upper_bound(all(c), (x)))
// functions
// gcd(0, x) fails.
ll gcd(ll a, ll b) {
    return b ? gcd(b, a % b) : a;
}
ll lcm(ll a, ll b) {
    return a / gcd(a, b) * b;
}
ll safemod(ll a, ll b) {
    return (a % b + b) % b;
}
template <class T> bool chmax(T &a, const T &b) {
    if (a < b) {
        a = b;
        return 1;
    }
    return 0;
}
template <class T> bool chmin(T &a, const T &b) {
    if (b < a) {
        a = b;
        return 1;
    }
    return 0;
}
template <typename T> T mypow(T x, ll n) {
    T ret = 1;
    while (n > 0) {
        if (n & 1)
            (ret *= x);
        (x *= x);
        n >>= 1;
    }
    return ret;
}
ll modpow(ll x, ll n, const ll mod) {
    ll ret = 1;
    while (n > 0) {
        if (n & 1)
            (ret *= x);
        (x *= x);
        n >>= 1;
        x %= mod;
        ret %= mod;
    }
    return ret;
}

uint64_t my_rand(void) {
    static uint64_t x = 88172645463325252ULL;
    x = x ^ (x << 13);
    x = x ^ (x >> 7);
    return x = x ^ (x << 17);
}
int popcnt(ull x) {
    return __builtin_popcountll(x);
}
template <typename T> vector<int> IOTA(vector<T> a) {
    int n = a.size();
    vector<int> id(n);
    iota(all(id), 0);
    sort(all(id), [&](int i, int j) { return a[i] < a[j]; });
    return id;
}
struct Timer {
    clock_t start_time;
    Timer() {
        start_time = clock();
    }
    void reset() {
        start_time = clock();
    }
    int lap() {
        // return x ms.
        return (clock() - start_time) * 1000 / CLOCKS_PER_SEC;
    }
};
template <int Mod> struct modint {
    int x;

    modint() : x(0) {
    }

    modint(long long y) : x(y >= 0 ? y % Mod : (Mod - (-y) % Mod) % Mod) {
    }

    modint &operator+=(const modint &p) {
        if ((x += p.x) >= Mod)
            x -= Mod;
        return *this;
    }

    modint &operator-=(const modint &p) {
        if ((x += Mod - p.x) >= Mod)
            x -= Mod;
        return *this;
    }

    modint &operator*=(const modint &p) {
        x = (int)(1LL * x * p.x % Mod);
        return *this;
    }

    modint &operator/=(const modint &p) {
        *this *= p.inverse();
        return *this;
    }

    modint operator-() const {
        return modint(-x);
    }

    modint operator+(const modint &p) const {
        return modint(*this) += p;
    }

    modint operator-(const modint &p) const {
        return modint(*this) -= p;
    }

    modint operator*(const modint &p) const {
        return modint(*this) *= p;
    }

    modint operator/(const modint &p) const {
        return modint(*this) /= p;
    }

    bool operator==(const modint &p) const {
        return x == p.x;
    }

    bool operator!=(const modint &p) const {
        return x != p.x;
    }

    modint inverse() const {
        int a = x, b = Mod, u = 1, v = 0, t;
        while (b > 0) {
            t = a / b;
            swap(a -= t * b, b);
            swap(u -= t * v, v);
        }
        return modint(u);
    }

    modint pow(int64_t n) const {
        modint ret(1), mul(x);
        while (n > 0) {
            if (n & 1)
                ret *= mul;
            mul *= mul;
            n >>= 1;
        }
        return ret;
    }

    friend ostream &operator<<(ostream &os, const modint &p) {
        return os << p.x;
    }

    friend istream &operator>>(istream &is, modint &a) {
        long long t;
        is >> t;
        a = modint<Mod>(t);
        return (is);
    }

    static int get_mod() {
        return Mod;
    }

    constexpr int get() const {
        return x;
    }
};
template <typename T = int> struct Edge {
    int from, to;
    T cost;
    int idx;

    Edge() = default;

    Edge(int from, int to, T cost = 1, int idx = -1) : from(from), to(to), cost(cost), idx(idx) {
    }

    operator int() const {
        return to;
    }
};

template <typename T = int> struct Graph {
    vector<vector<Edge<T>>> g;
    int es;

    Graph() = default;

    explicit Graph(int n) : g(n), es(0) {
    }

    size_t size() const {
        return g.size();
    }

    void add_directed_edge(int from, int to, T cost = 1) {
        g[from].emplace_back(from, to, cost, es++);
    }

    void add_edge(int from, int to, T cost = 1) {
        g[from].emplace_back(from, to, cost, es);
        g[to].emplace_back(to, from, cost, es++);
    }

    void read(int M, int padding = -1, bool weighted = false, bool directed = false) {
        for (int i = 0; i < M; i++) {
            int a, b;
            cin >> a >> b;
            a += padding;
            b += padding;
            T c = T(1);
            if (weighted)
                cin >> c;
            if (directed)
                add_directed_edge(a, b, c);
            else
                add_edge(a, b, c);
        }
    }
};

/* #endregion*/
// constant
#define inf 1000000000ll
#define INF 4000000004000000000LL

long long xor64(long long range) {
    static uint64_t x = 88172645463325252ULL;
    x ^= x << 13;
    x ^= x >> 7;
    return (x ^= x << 17) % range;
}

/* #region min cost flow */
template <typename flow_t, typename cost_t>
struct MinCostFlow {
    const cost_t TINF;
    struct edge {
        int to;
        flow_t cap;
        cost_t cost;
        int rev;
        bool isrev;
    };
    vector<vector<edge> > graph;
    vector<cost_t> potential, min_cost;
    vector<int> prevv, preve;

    MinCostFlow(int V) : TINF(numeric_limits<cost_t>::max()), graph(V) {}

    void add_edge(int from, int to, flow_t cap, cost_t cost) {
        graph[from].emplace_back(
            (edge){to, cap, cost, (int)graph[to].size(), false});
        graph[to].emplace_back(
            (edge){from, 0, -cost, (int)graph[from].size() - 1, true});
    }

    cost_t min_cost_flow(int s, int t, flow_t f) {
        int V = (int)graph.size();
        cost_t ret = 0;
        using Pi = pair<cost_t, int>;
        priority_queue<Pi, vector<Pi>, greater<Pi> > que;
        potential.assign(V, 0);
        preve.assign(V, -1);
        prevv.assign(V, -1);

        while (f > 0) {
            min_cost.assign(V, TINF);
            que.emplace(0, s);
            min_cost[s] = 0;
            while (!que.empty()) {
                Pi p = que.top();
                que.pop();
                if (min_cost[p.second] < p.first) continue;
                for (int i = 0; i < graph[p.second].size(); i++) {
                    edge &e = graph[p.second][i];
                    cost_t nextCost = min_cost[p.second] + e.cost +
                                      potential[p.second] - potential[e.to];
                    if (e.cap > 0 && min_cost[e.to] > nextCost) {
                        min_cost[e.to] = nextCost;
                        prevv[e.to] = p.second, preve[e.to] = i;
                        que.emplace(min_cost[e.to], e.to);
                    }
                }
            }
            if (min_cost[t] == TINF) return -1;
            for (int v = 0; v < V; v++) potential[v] += min_cost[v];
            flow_t addflow = f;
            for (int v = t; v != s; v = prevv[v]) {
                addflow = min(addflow, graph[prevv[v]][preve[v]].cap);
            }
            f -= addflow;
            ret += addflow * potential[t];
            for (int v = t; v != s; v = prevv[v]) {
                edge &e = graph[prevv[v]][preve[v]];
                e.cap -= addflow;
                graph[v][e.rev].cap += addflow;
            }
        }
        return ret;
    }

    void output() {
        for (int i = 0; i < graph.size(); i++) {
            for (auto &e : graph[i]) {
                if (e.isrev) continue;
                auto &rev_e = graph[e.to][e.rev];
                cout << i << "->" << e.to << " (flow: " << rev_e.cap << "/"
                     << rev_e.cap + e.cap << ")" << endl;
            }
        }
    }
};
/* #endregion*/

int min_tmp = 0;
int max_tmp = 1000;
int max_measurements = 10000;
const vector<int> dx = {0, 0, 1, 0, -1, 1, 1, -1, -1, 2, 0, -2, 0, 2, -2, 2, -2};
const vector<int> dy = {0, 1, 0, -1, 0, 1, -1, 1, -1, 0, 2, 0, -2, 2, 2, -2, -2};

vi X, Y;
int N, L, S;

double measure(int i, int x, int y){
    cout << i << ' ' << x << ' ' << y << endl;
    double T;
    cin >> T;
    return T;
}
template<typename T>
void debug(T x){
    cerr << "# " << x << endl;
}
int nml(int x){
    return (x % L + L) % L;
}


struct Change1{
    int x, y, dP;
};


struct State1{
    ll score;
    mat<ll> P;
    set<pair<int, int>> fixed_cells;
    State1(mat<ll> P, set<pair<int, int>> fixed_cells) : P(P), fixed_cells(fixed_cells){
        score = 0;
        rep(i, L){
            rep(j, L){
                score += mypow<ll>(P[i][(j + 1) % L] - P[i][j], 2);
                score += mypow<ll>(P[(i + 1) % L][j] - P[i][j], 2);
            }
        }
    }

    Change1 generate_change(){
        int x = xor64(P.size());
        int y = xor64(P.size());
        while(fixed_cells.count(mp(x, y))){
            x = xor64(P.size());
            y = xor64(P.size());
        }
        int dP = (P[nml(x + 1)][nml(y)] + P[nml(x - 1)][nml(y)] + P[nml(x)][nml(y + 1)] + P[nml(x)][nml(y - 1)]) / 4 - P[x][y];
        if(P[x][y] + dP < min_tmp || P[x][y] + dP > max_tmp) dP *= -1;
        return {x, y, dP};
    }

    ll get_new_score(Change1 c){
        int x = c.x, y = c.y, dP = c.dP;
        ll new_score = score;
        new_score -= mypow<ll>(P[x][(y + 1) % L] - P[x][y], 2);
        new_score -= mypow<ll>(P[(x + 1) % L][y] - P[x][y], 2);
        new_score -= mypow<ll>(P[x][(y - 1 + L) % L] - P[x][y], 2);
        new_score -= mypow<ll>(P[(x - 1 + L) % L][y] - P[x][y], 2);
        P[x][y] += dP;
        new_score += mypow<ll>(P[x][(y + 1) % L] - P[x][y], 2);
        new_score += mypow<ll>(P[(x + 1) % L][y] - P[x][y], 2);
        new_score += mypow<ll>(P[x][(y - 1 + L) % L] - P[x][y], 2);
        new_score += mypow<ll>(P[(x - 1 + L) % L][y] - P[x][y], 2);
        P[x][y] -= dP;
        return new_score;
    }  

    void step(Change1 c, ll new_score){
        int x = c.x, y = c.y, dP = c.dP;
        P[x][y] += dP;
        score = new_score;
    } // 実際の更新

    bool operator<(const State1 &rhs) const {
        return score < rhs.score;
    }
};


struct Change2{
    int x, y, dP;
};


struct State2{
    double score;
    int L;
    mat<ll> P;
    int measure_cell;
    int min_i;

    ll calc_dist(int i, int j){
        ll dist = 0;
        rep(k, measure_cell){
            int xi = nml(X[i] + dx[k]);
            int yi = nml(Y[i] + dy[k]);
            int xj = nml(X[j] + dx[k]); 
            int yj = nml(Y[j] + dy[k]);
            dist += (P[xi][yi] - P[xj][yj]) * (P[xi][yi] - P[xj][yj]);
        }
        return dist;
    }

    double calc_score(){
        vl dists;
        ll min_dist = INF;
        ll cnt = 0;
        rep(i, N){
            rep(j, i+1, N){
                ll dist = calc_dist(i, j);
                dists.pb(dist);
                if(chmin(min_dist, dist)) min_i = i;
            }
        }
        for(ll dist: dists){
            if(dist == min_dist) cnt++;
        }
        return -min_dist * 1000 + cnt;
    }

    State2(mat<ll> P, int measure_cell) : P(P), measure_cell(measure_cell){
        score = calc_score();
    }

    Change2 generate_change(){
        int i = min_i;
        int k = xor64(measure_cell);
        int x = nml(X[i] + dx[k]);
        int y = nml(Y[i] + dy[k]);
        int dP = xor64(max_tmp) - max_tmp / 2;
        if(P[x][y] + dP < min_tmp || P[x][y] + dP > max_tmp) dP *= -1;
        return {x, y, dP};
    }

    double get_new_score(Change2 c){
        int x = c.x, y = c.y, dP = c.dP;
        P[x][y] += dP;
        double new_score = calc_score();
        P[x][y] -= dP;
        return new_score;
    }  

    void step(Change2 c, double new_score){
        int x = c.x, y = c.y, dP = c.dP;
        P[x][y] += dP;
        score = new_score;
    } // 実際の更新
};


template<typename State, typename Change>
State hill_climbing(State state, double max_time){
    Timer timer;
    while (timer.lap() < max_time) {
        double score = state.score;
        Change change = state.generate_change();
        double new_score = state.get_new_score(change);
        if (new_score < score) {
            state.step(change, new_score);
        }
    }
    return state;
}

int main(int argc, char *argv[]) {
    cin.tie(0);
    ios::sync_with_stdio(0);
    cout << setprecision(30) << fixed;
    cerr << setprecision(30) << fixed;

    // 入力
    cin >> L >> N >> S;
    X.resize(N), Y.resize(N);
    rep(i, N) {
        cin >> X[i] >> Y[i];
    }

    // 配置
    int measure_cells = 5;
    if(S >= 100) measure_cells = 9;
    if(S >= 750) measure_cells = 13;
    mat<int> true_tmps(N, vi(measure_cells, 0));
    mat<ll> P(L, vl(L, 0));
    set<pair<int, int>> fixed_cells;
    rep(x, L){
        rep(y, L){
            if(S <= 750){
                P[x][y] = xor64(max_tmp);
            }else{
                P[x][y] = xor64(2) * max_tmp;
            }
        }
    }
    State2 state2(P, measure_cells);
    state2 = hill_climbing<State2, Change2>(state2, 1500);
    P = state2.P;

    rep(i, N){
        rep(j, measure_cells){
            int x = nml(X[i] + dx[j]);
            int y = nml(Y[i] + dy[j]);
            true_tmps[i][j] = P[x][y];
            fixed_cells.insert(mp(x, y));
        }
    }
    rep(x, L){
        rep(y, L){
            if(fixed_cells.count(mp(x, y))) continue;
            double normalization = 0;
            double sum_ = 0;
            rep(i, N){
                rep(j, measure_cells){
                    int nx = nml(X[i] + dx[j]);
                    int ny = nml(Y[i] + dy[j]);
                    double weight = exp(-0.1 * (abs(x - X[i]) + abs(y - Y[i])));
                    sum_ += weight * P[nx][ny];
                    normalization += weight;
                }
            }
            P[x][y] = sum_ / normalization;
        }
    }
    State1 state(P, fixed_cells);
    if(fixed_cells.size() < L * L) state = hill_climbing<State1, Change1>(state, 100);

    double min_dist = INF;
    rep(i, N){
        rep(j, i + 1, N){
            double dist = 0;
            rep(k, measure_cells){
                dist += mypow<ll>(true_tmps[i][k] - true_tmps[j][k], 2);
            }
            chmin(min_dist, dist);
        }
    }

    int measure_times = max_measurements / N / measure_cells;
    double placement_cost = state.score;
    double measurement_cost = measure_times * N * measure_cells * 1000;
    double level = 15;
    
    // 最適化
    double measure_coef = sqrt(level * placement_cost * S * S / min_dist / measure_times / measurement_cost);
    chmax(measure_coef, level * S * S / min_dist / measure_times);
    chmin(measure_coef, 1.0);
    double tmp_coef = level * S * S / min_dist / measure_times / measure_coef;
    chmin(tmp_coef, 1.0);

    measure_times = (double)measure_times * measure_coef;
    chmax(measure_times, 1);

    rep(i, L) {
        rep(j, L){
            state.P[i][j] = (double)state.P[i][j] * sqrt(tmp_coef) + (1 - sqrt(tmp_coef)) * max_tmp / 2;
            cout << state.P[i][j] << ' ';
        }
        cout << endl;
    }
    rep(i, N){
        rep(j, measure_cells){
            int x = nml(X[i] + dx[j]);
            int y = nml(Y[i] + dy[j]);
            true_tmps[i][j] = state.P[x][y];
            fixed_cells.insert(mp(x, y));
        }
    }

    //計測
    mat<double> estimated_tmps(N, vector<double>(measure_cells, 0));
    rep(i, N){
        rep(j, measure_cells){
            rep(_, measure_times){
                estimated_tmps[i][j] += measure(i, dx[j], dy[j]) / measure_times;   
            }
        }
    }
    cout << -1 << ' ' << -1 << ' ' << -1 << endl;

    // 回答
    MinCostFlow<int, ll> mcf(N * 2 + 2);
    rep(i, N){
        mcf.add_edge(N * 2, i, 1, 0);
        mcf.add_edge(N + i, N * 2 + 1, 1, 0);
        rep(j, N){
            double cost = 0;
            rep(k, measure_cells){
                cost += mypow<double>(estimated_tmps[i][k] - true_tmps[j][k], 2); 
            }
            mcf.add_edge(i, N + j, 1, cost);
        }
    }
    mcf.min_cost_flow(N * 2, N * 2 + 1, N);

    vi E(N);
    rep(i, N){
        for(auto& e : mcf.graph[i]){
            if (e.isrev) continue;
            auto &rev_e = mcf.graph[e.to][e.rev];
            if(rev_e.cap == 1){
                E[i] = e.to - N;
            }
        }
    }
    rep(i, N){
        cout << E[i] << endl;
    }
}