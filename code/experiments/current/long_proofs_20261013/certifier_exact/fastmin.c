// Exact minimal-proof-length search for the ND take-home logic: the costs of vendor/fork/minlen.py, in C.
//
// minlen computes best(G, A, b) = m(G, A) if m(G, A) <= b else None, where m(G, A) is the least cost of a
// derivation tree of G from the citable set A in minlen's space (each sub-call's budget strictly decreases, so
// the recursion is well founded and m is well defined; branch-and-bound only drops candidates that cannot beat
// the incumbent). This file computes the same m at the top level (conclusion, premises). Differences:
//   * memo keyed by (G, A) not (G, A, b): an entry holds the exact m or a lower bound "m > lb", so a query with
//     budget b <= lb fails at once and iterative deepening reuses everything below the new bound.
//   * the memo is a bounded cache: one 64-byte record per citable set A with a byte per goal (|S| <= 55; 128 B for
//     |S| <= 111), two-choice hashing, doubling up to the memory cap, then eviction. Losing an entry only costs
//     recomputation. search() reads its own A's record through a cached pointer (most queries share the parent's A).
//   * semantic pruning: if A does not entail G (truth table over the theorem's atoms) then m(G, A) = inf,
//     because every derivation in the space is a sound ND derivation of G from A.
//   * budget 1 is answered by bit tests (fast1) instead of a search: m <= 1 iff a one-step rule has all inputs in A.
//   * REDUCED RULES (search_reduced; -DFULLRULES disables): a one-step rule or ORE whose input X is derived costs
//     m(X, A) + rest + 1, and LEMMA(X) then the same rule with X in A costs m(X, A) + m(G, A+X) <= that. So the
//     search tries only: one-step rules with every input in A (cost 1), IMPI, NEGI, RAA, ORE over D in A, LEMMA.
//   * LEMMA HOISTING (-DNOHOIST disables): a lemma's own derivation never needs a LEMMA at its top, so the lemma
//     loop uses m-(L, A) (no top lemma; memo kind 1) for L and the full m for the continuation. Proof: among
//     optimal derivations of (G, A) starting with a lemma, take one whose lemma L has the least m(L, A). If L's
//     derivation started with lemma L', then m(L', A) + m(G, A+L') <= m(L', A) + m(L, A+L') + m(G, A+L+L')
//     <= m(L, A) + m(G, A+L), and m(L', A) < m(L, A) (L' != G, else m(G, A) < m(G, A)): a contradiction.
//     Proof templates are rebuilt with the full rule set (search() with a choice), which has the same minimum.
//   * NO TOWERS (default; build with -DTOWERS for minlen's literal rule): DN on goal G is tried only when ~~G
//     is in S. minlen also tries DN when ~~G is outside S, which creates goals ~~~~x, ~~~~~~x, ... and NEGI
//     hypotheses outside S; that is most of minlen's search (9x the calls on our bench) and never shortens a
//     proof. Proof below. With it every goal and every member of A lies in S, so A is a bitset over S.
//     The -DTOWERS build keeps minlen's DN literally (and the older (G, A)-keyed table) for cross-checking.
//
// PROOF THAT "NO TOWERS" PRESERVES THE COST. Notation: S = sub u ~sub u ~~sub u {F} (minlen's S); m_f = minlen's
// cost (full system), m_r = cost with DN restricted to ~~G in S. Write best(G,A) for the cost of a derivation of
// goal G (1 if G in A) and inp(G,A) for an input (0 if G in A). Restricted derivations are full derivations, so
// m_r >= m_f. For <=: every formula the full search touches is ~^n s with s in S. Define phi(f) = f for f in S;
// otherwise strip leading ~~ until the result is in S, except that ~F (when F not in sub, so ~F not in S) maps to
// DROP (it is a tautology and only ever cites as NEGE(F, ~F), which derives F from F). phi(A) = images minus DROP.
// Facts: (F1) phi(A u {f}) = phi(A) u {phi f}; f not in S => f = ~~h with phi f = phi h, or f is an odd F-tower.
// (F2) G not in S => phi G not in sub (else ~~phi G in S would have stopped the stripping): phi G is a not or F.
// (F3) ~^k g in S with k >= 2 => ~^(k-2) g in sub. (F4) for f with phi f = g in S: if ~g in S then
// phi(~f) = ~g; else g = ~~y with y in sub, ~y not in sub, and phi(~f) = ~y, i.e. g = ~phi(~f) (by F3 the
// stripping of ~f cannot stop early). Odd F-towers are never goals (no rule makes ~F a goal).
// Lemma B (restricted system, A in S): let z in sub with ~z not in sub (so ~~z is not in sub and ~~~z is not in
// S); if ~~z in A and G != ~~z then best_r(G, A - {~~z} + {z}) <= best_r(G, A). Induction on cost. ~~z is not a
// component of any formula of S and has no NEGE partner ~~~z, so with ~~z in A a child goal ~~z only comes from
// DN for goal z (now z in A: cost 1 / 0) or NEGE(X=~z, NX=~~z) (variant "NX in A", cost inp(~z)+1; now X=z in A,
// NX=~z in S: cost inp(~z)+1 or 1). Everything else maps rule for rule (lemma L=z: z is now in A, use IH on the
// continuation; a rule adding ~~z adds nothing to A and its image is a superset, fine by antitonicity); NEGE
// variants only improve when membership grows. (With only "~~z not in sub" it fails: {~~z, ~~~z} gives F in one
// line, {z, ~~~z} needs two.) best_r is antitone in A (more citable formulas never cost more), same induction.
// Main lemma: for every G, A the full search reaches, best_r(phi G, phi A) <= best_f(G, A). Induction on the
// full cost c; if phi G in phi A the left side is 1. Otherwise take the rule of the optimal full derivation:
//  G in S: ANDE, IMPE, ANDI, ORI, IMPI, NEGI, RAA, BOTE, ORE, LEMMA map rule for rule (their formulas are in S;
//   lemma L with L in phi A: use IH on the continuation). DN with ~~G in S maps to itself; with ~~G not in S,
//   phi(~~G) = G so best_r(G, phi A) <= inp_f(~~G, A) = c-1. NEGE(X, NX): NX in S maps (membership only grows);
//   NX not in S (so NX in A): X = F gives cost inp_f(F, A)+1 for goal F, dominated by its own child; X = ~~y gives
//   phi(NX) = ~y in phi A, and the restricted NEGE(X'=~y, NX'=~~y) costs inp_r(~~y)+1 <= c.
//  G not in S (G = ~~h, a not outside sub): only R, DN, BOTE, NEGI, ORE, LEMMA apply. DN: as above. BOTE: phi G
//   = F => child bound; else BOTE. ORE: disjuncts are in sub and phi G is not (F2), so branches map. LEMMA: as above.
//   NEGI (hyp Y, G = ~Y, c = best_f(F, A+Y) + 2; IH: best_r(F, phi A + phi Y) <= c-2): by F4 on Y, either
//   phi G = ~phi Y and the restricted NEGI on phi G uses exactly that hypothesis; or phi Y = ~phi G = ~~Z with
//   phi G = ~Z (F2: outside sub, so ~Z not in sub; Z in sub by F3) and Lemma B gives best_r(F, phi A + Z)
//   <= best_r(F, phi A + ~~Z), so the restricted NEGI on ~Z costs <= c; or phi Y = DROP, phi G = F, and IH bounds
//   best_r(F, phi A) by c-2.
// At the top level phi is the identity (conclusion and premises are in S), so m_r = m_f. Checked empirically
// too: identical labels on 539 minlen-labelled records and on random sequents with F and ~~ (fuzz_tower*.py).
//
// Build: clang -O3 -shared -fPIC -DNW=1 fastmin.c -o libfastmin1.dylib   (NW = 64-bit words in A; certify_fast.py
// builds and picks the width).
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <setjmp.h>
#include <time.h>

#ifndef NW
#define NW 2
#endif
#ifndef MEMO_MIN_B
#define MEMO_MIN_B 2
#endif
#define MAXF 16384
#define TTW_MAX 16
#define INF 10000

enum { K_ATOM = 0, K_BOT = 1, K_NOT = 2, K_AND = 3, K_OR = 4, K_IMP = 5 };
enum { R_R = 0, R_ANDE1, R_ANDE2, R_IMPE, R_DN, R_BOTE, R_ANDI, R_ORI1, R_ORI2, R_IMPI, R_NEGI, R_NEGE, R_RAA, R_ORE,
       R_LEMMA, R_NONE };
static const char *RN[] = {"R", "ANDE1", "ANDE2", "IMPE", "DN", "BOTE", "ANDI", "ORI1", "ORI2", "IMPI", "NEGI",
                           "NEGE", "RAA", "ORE", "LEMMA"};

typedef struct { uint64_t w[NW]; } aset;
typedef struct { uint64_t w[NW]; uint32_t g; uint8_t val; uint8_t pad[3]; } entry;  // val: 0 empty, 128|v exact, lb

static int nf, nf0, BOTID;
static int kind[MAXF], fl[MAXF], fr[MAXF], negid[MAXF], inS[MAXF], abit[MAXF];
static uint64_t tt[MAXF][TTW_MAX];
static int ttw;
static uint64_t fullmask_last;
static int nabits;
static int Sl[MAXF], nS, Dl[MAXF], nD;
// per-goal lists (pre-registered formulas only): conjunctions with G as a component, implications with consequent G
static int ande_off[MAXF + 1], ande_c[4 * MAXF], ande_w[4 * MAXF];
static int impe_off[MAXF + 1], impe_i[4 * MAXF];

static uint64_t tabmask, cap_max;  // table size - 1; entries allowed by the memory cap
static int64_t n_used, n_vals;  // records (or entries) in use; stored (G, A) values
static jmp_buf jb;
static double deadline;
static int64_t st_calls, st_sem, st_hit, st_store, st_evict;

static double now(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

static int neg(int f) {
  if (negid[f] >= 0) return negid[f];
#ifndef TOWERS
  longjmp(jb, 9);  // unreachable: without towers every formula used is in S
#endif
  if (nf >= MAXF) longjmp(jb, 3);
  int id = nf++;
  kind[id] = K_NOT; fl[id] = f; fr[id] = -1; negid[id] = -1; inS[id] = 0; abit[id] = -1;
  for (int i = 0; i < ttw; i++) tt[id][i] = ~tt[f][i];
  tt[id][ttw - 1] &= fullmask_last;
  negid[f] = id;
  return id;
}

static inline int inA(const aset *A, int f) {
  int b = abit[f];
  return b >= 0 && ((A->w[b >> 6] >> (b & 63)) & 1);
}

static inline void addA(aset *A2, uint64_t *t2, const aset *A, const uint64_t *tA, int f) {
  *A2 = *A;
  int b = abit[f];
  if (b < 0) {
    if (nabits >= NW * 64) longjmp(jb, 2);
    b = abit[f] = nabits++;
  }
  A2->w[b >> 6] |= 1ULL << (b & 63);
  for (int i = 0; i < ttw; i++) t2[i] = tA[i] & tt[f][i];
}

static inline int entails(const uint64_t *tA, int G) {
  for (int i = 0; i < ttw; i++)
    if (tA[i] & ~tt[G][i]) return 0;
  return 1;
}

static inline uint64_t fmix64(uint64_t h) {
  h ^= h >> 33; h *= 0xff51afd7ed558ccdULL;
  h ^= h >> 33; h *= 0xc4ceb9fe1a85ec53ULL;
  h ^= h >> 33;
  return h;
}

#ifdef TOWERS
#define SLOT_T entry
static entry *tab;
static inline uint64_t hkey(int G, const aset *A) {
  uint64_t h = fmix64((uint64_t)G + 0x9E3779B97F4A7C15ULL);
  for (int i = 0; i < NW; i++) h = fmix64(h ^ A->w[i]);
  return h;
}

// Memo: two-choice bucketed hash table. Each key may live in one of two 4-entry buckets (64 B for NW=1); when both
// are full the table doubles (up to the memory cap) or, at the cap, evicts the entry with the smallest budget.
#define BUCKET 4
static inline int keyeq(const entry *e, int G, const aset *A) {
  if (e->g != (uint32_t)G) return 0;
  for (int i = 0; i < NW; i++)
    if (e->w[i] != A->w[i]) return 0;
  return 1;
}

static inline void buckets(int G, const aset *A, uint64_t *b1, uint64_t *b2) {
  uint64_t h = hkey(G, A);
  uint64_t nb = (tabmask + 1) / BUCKET;
  *b1 = (h % nb) * BUCKET;
  *b2 = (fmix64(h ^ 0x5bd1e9955bd1e995ULL) % nb) * BUCKET;
}

static entry *lookup(int G, const aset *A) {
  uint64_t b1, b2;
  buckets(G, A, &b1, &b2);
  for (int k = 0; k < BUCKET; k++) {
    entry *e = &tab[b1 + k];
    if (e->val && keyeq(e, G, A)) return e;
  }
  for (int k = 0; k < BUCKET; k++) {
    entry *e = &tab[b2 + k];
    if (e->val && keyeq(e, G, A)) return e;
  }
  return NULL;
}

static void grow(void);

static void store(int G, const aset *A, uint8_t val) {
  uint64_t b1, b2;
  buckets(G, A, &b1, &b2);
  entry *victim = NULL, *empty = NULL;
  for (int k = 0; k < 2 * BUCKET; k++) {
    entry *e = &tab[(k < BUCKET ? b1 : b2 - BUCKET) + k];
    if (!e->val) { if (!empty) empty = e; continue; }
    if (keyeq(e, G, A)) {
      if (!(e->val & 128)) e->val = val;  // exact never downgraded; lb only grows (searched only when lb < b)
      return;
    }
    if (!victim || (e->val & 127) < (victim->val & 127)) victim = e;
  }
  if (empty) { victim = empty; n_used++; }
  else {
    // both buckets full: grow if the table is half full and the cap allows, else evict (rare below that load)
    if ((uint64_t)n_used * 2 > tabmask + 1 && (tabmask + 1) * 2 <= cap_max) { grow(); store(G, A, val); return; }
    st_evict++;
  }
  st_store++;
  for (int i = 0; i < NW; i++) victim->w[i] = A->w[i];
  victim->g = G;
  victim->val = val;
}

static void grow(void) {
  uint64_t ocap = tabmask + 1, ncap = ocap * 2;
  entry *old = tab, *nt = calloc(ncap, sizeof(entry));
  if (!nt) { cap_max = ocap; return; }
  tab = nt; tabmask = ncap - 1; n_used = 0;
  for (uint64_t i = 0; i < ocap; i++)
    if (old[i].val) store(old[i].g, (const aset *)old[i].w, old[i].val);
  free(old);
}

static inline uint8_t mget(int G, const aset *A, int kd) {
  (void)kd;
  entry *e = lookup(G, A);
  return e ? e->val : 0;
}
static inline void mput(int G, const aset *A, int kd, uint8_t val) { (void)kd; store(G, A, val); }
#else
// Memo: one record per citable set A holding a byte per goal (goals are S formulas, ids 0..|S|-1): 0 unknown,
// 128|v exact, else a lower bound "m > val". Queries for the same A (lemma candidates, rule inputs) share a cache
// line. Two-choice hashing over 2-record buckets; when all 4 candidate records are taken the table doubles (if a
// quarter full and under the memory cap) or evicts the record with the smallest largest budget.
// With lemma hoisting (default) a record holds two bytes per goal: kind 0 = m(G, A), kind 1 = m-(G, A), the cost
// without a LEMMA as the top rule (what a lemma's own derivation needs, see search_reduced).
#ifdef NOHOIST
#define NKIND 1
#else
#define NKIND 2
#endif
#define RECB (64 * NW * NKIND)
#define NSLOT ((RECB - 8 * NW - 1) / NKIND)   // goals per record: 59 (NW=1), 119 (NW=2), 239 (NW=4)
typedef struct { uint64_t w[NW]; uint8_t prio; uint8_t v[NKIND * NSLOT]; } rec;   // 128 B (NW=1), 256 B (NW=2)
#define SLOT_T rec
#define RB 2
static rec *tab;

static inline void rbuckets(const aset *A, uint64_t *b1, uint64_t *b2) {
  uint64_t h = 0x9E3779B97F4A7C15ULL;
  for (int i = 0; i < NW; i++) h = fmix64(h ^ A->w[i]);
  uint64_t nb = (tabmask + 1) / RB;
  *b1 = (h % nb) * RB;
  *b2 = (fmix64(h ^ 0x5bd1e9955bd1e995ULL) % nb) * RB;
}
static inline int aeq(const rec *r, const aset *A) {
  for (int i = 0; i < NW; i++)
    if (r->w[i] != A->w[i]) return 0;
  return 1;
}
static rec *findrec(const aset *A) {
  uint64_t b1, b2;
  rbuckets(A, &b1, &b2);
  for (int k = 0; k < RB; k++) { rec *r = &tab[b1 + k]; if (r->prio && aeq(r, A)) return r; }
  for (int k = 0; k < RB; k++) { rec *r = &tab[b2 + k]; if (r->prio && aeq(r, A)) return r; }
  return NULL;
}
static inline uint8_t mget(int G, const aset *A, int kd) {
  rec *r = findrec(A);
  return r ? r->v[kd * NSLOT + G] : 0;
}
static void grow(void);
static uint64_t epoch;  // bumped whenever records are created, evicted or moved: cached rec pointers are stale
static rec *place(const aset *A) {  // a cleared record for A (not present)
  epoch++;
  uint64_t b1, b2;
  rbuckets(A, &b1, &b2);
  rec *victim = NULL, *empty = NULL;
  for (int k = 0; k < 2 * RB; k++) {
    rec *r = &tab[k < RB ? b1 + k : b2 + k - RB];
    if (!r->prio) { if (!empty) empty = r; continue; }
    if (!victim || r->prio < victim->prio) victim = r;
  }
  if (empty) { victim = empty; n_used++; }
  else {
    if ((uint64_t)n_used * 4 > tabmask + 1 && (tabmask + 1) * 2 <= cap_max) { grow(); return place(A); }
    st_evict++;
    for (int i = 0; i < NKIND * NSLOT; i++) n_vals -= victim->v[i] != 0;
  }
  memset(victim, 0, sizeof(rec));
  for (int i = 0; i < NW; i++) victim->w[i] = A->w[i];
  return victim;
}
static void mput(int G, const aset *A, int kd, uint8_t val) {
  rec *r = findrec(A);
  int i = kd * NSLOT + G;
  if (!r) r = place(A);
  else if (r->v[i] & 128) return;  // exact never downgraded; lb only grows (searched only when lb < b)
  n_vals += !r->v[i];
  r->v[i] = val;
  uint8_t p = val & 127;
  if (p < 1) p = 1;
  if (p > r->prio) r->prio = p;
  st_store++;
}
static void grow(void) {
  epoch++;
  uint64_t ocap = tabmask + 1, ncap = ocap * 2;
  rec *old = tab, *nt = calloc(ncap, sizeof(rec));
  if (!nt) { cap_max = ocap; return; }
  tab = nt; tabmask = ncap - 1; n_used = 0;
  for (uint64_t i = 0; i < ocap; i++)
    if (old[i].prio) { rec *r = place((const aset *)old[i].w); memcpy(r, &old[i], sizeof(rec)); }
  free(old);
}
#endif

typedef struct { int rule, p; } choice;
static int search(int G, const aset *A, const uint64_t *tA, int b, choice *ch);

#ifndef TOWERS
// Budget 1 without recursion (abit[f] == f for every formula): best(G, A, 1) = 1 iff G in A or some rule's inputs
// are all in A: ANDE (C in A), IMPE (I and I1 in A), DN (~~G in A), BOTE (F in A), ANDI (G1 in A and G2 in A + G1),
// ORI (G1 or G2 in A), NEGE for G = F (X and ~X in A). IMPI/NEGI/RAA/ORE/LEMMA need budget >= 2. Same as search(G,A,1).
static aset m_ande[MAXF];
static aset true_at[1 << 10];   // for each valuation row, the S formulas true there
static aset all_S;
#ifndef FULLRULES
// E = the S formulas entailed by A: AND of true_at over the rows satisfying A (all of S if A is unsatisfiable)
static inline void entailed_set(aset *E, const uint64_t *tA) {
  *E = all_S;
  for (int w = 0; w < ttw; w++)
    for (uint64_t x = tA[w]; x; x &= x - 1) {
      const aset *t = &true_at[w * 64 + __builtin_ctzll(x)];
      for (int i = 0; i < NW; i++) E->w[i] &= t->w[i];
    }
}
#endif
static int ip_off[MAXF + 1], ip_i[4 * MAXF];
static int np_x[MAXF], np_nx[MAXF], n_np;
static inline int bit(const aset *A, int f) { return (A->w[f >> 6] >> (f & 63)) & 1; }
static int fast1(int G, const aset *A) {
  // m_ande[G] = every formula whose presence alone gives cost 1: G, conjunctions with component G, ~~G (if in S),
  // F (if G != F), and G's disjuncts (if G is a disjunction)
  for (int i = 0; i < NW; i++) if (A->w[i] & m_ande[G].w[i]) return 1;
  for (int j = ip_off[G]; j < ip_off[G + 1]; j++) { int I = ip_i[j]; if (bit(A, I) && bit(A, fl[I])) return 1; }
  if (kind[G] == K_AND && bit(A, fl[G]) && (bit(A, fr[G]) || fr[G] == fl[G])) return 1;
  if (G == BOTID)
    for (int j = 0; j < n_np; j++) if (bit(A, np_x[j]) && bit(A, np_nx[j])) return 1;
  return INF;
}
#endif

static int best(int G, const aset *A, const uint64_t *tA, int b) {
  if (b <= 0) return INF;
  if (inA(A, G)) return 1;
  if (!entails(tA, G)) { st_sem++; return INF; }
#ifndef TOWERS
  if (b == 1) return fast1(G, A);
#endif
  if (b < MEMO_MIN_B) return search(G, A, tA, b, NULL);
  uint8_t v = mget(G, A, 0);
  if (v) {
    st_hit++;
    if (v & 128) { v &= 127; return v <= b ? v : INF; }
    if (v >= b) return INF;
  }
  int r = search(G, A, tA, b, NULL);
  mput(G, A, 0, r <= b ? (uint8_t)(128 | r) : (uint8_t)b);
  return r;
}

static inline int inp(int G, const aset *A, const uint64_t *tA, int b) {
  if (inA(A, G)) return 0;
  return best(G, A, tA, b);
}

#ifndef TOWERS
// best() for a goal on the caller's own A, reading the memo through the caller's cached record pointer (refetched
// only when `epoch` moved); same result as best().
// kd = 0: m(G, A); kd = 1: m-(G, A), no LEMMA at the top (search_reduced with lem = 0).
typedef struct { rec *r; uint64_t ep; } curA;
#ifndef FULLRULES
static int search_reduced(int G, const aset *A, const uint64_t *tA, int b, int lem);
#endif
static inline int best_c(int G, const aset *A, const uint64_t *tA, int b, curA *c, int kd) {
  if (b <= 0) return INF;
  if (inA(A, G)) return 1;
  if (!entails(tA, G)) { st_sem++; return INF; }
  if (b == 1) return fast1(G, A);
  if (c->ep != epoch || !c->r) { c->r = findrec(A); c->ep = epoch; }
  uint8_t v = c->r ? c->r->v[kd * NSLOT + G] : 0;
  if (v) {
    st_hit++;
    if (v & 128) { v &= 127; return v <= b ? v : INF; }
    if (v >= b) return INF;
  }
#ifdef FULLRULES
  int r = search(G, A, tA, b, NULL);
#else
  int r = kd ? search_reduced(G, A, tA, b, 0) : search(G, A, tA, b, NULL);
#endif
  mput(G, A, kd, r <= b ? (uint8_t)(128 | r) : (uint8_t)b);
  return r;
}
static inline int inp_c(int G, const aset *A, const uint64_t *tA, int b, curA *c) {
  if (inA(A, G)) return 0;
  return best_c(G, A, tA, b, c, 0);
}
#define INPA(G, B) inp_c(G, A, tA, B, &cur)
#define BESTA(G, B) best_c(G, A, tA, B, &cur, 0)
#else
#define INPA(G, B) inp(G, A, tA, B)
#define BESTA(G, B) best(G, A, tA, B)
#endif

#define LIM (bestc - 1)
#define CONSIDER(c, R, P) do { int _c = (c); if (_c < bestc) { bestc = _c; if (ch) { ch->rule = (R); ch->p = (P); } } } while (0)

#if !defined(TOWERS) && !defined(FULLRULES)
// Reduced rule set, same cost (used when no proof template is needed). Every one-step rule (ANDE, IMPE, DN, BOTE,
// ANDI, ORI, NEGE) and ORE's disjunction input that DERIVES an input X (X not in A) costs m(X, A) + rest + 1, and
// LEMMA(X) followed by the same rule with X in A costs m(X, A) + m(G, A+X) <= m(X, A) + rest + 1 (X is in S, not
// in A, and != G; the cases X = G -- NEGE(F, ~F) for goal F, IMPE with antecedent G -- are never optimal). So it is
// enough to try: every one-step rule with all inputs in A (fast1: cost 1), IMPI, NEGI, RAA, ORE over D in A, LEMMA.
// lem = 0 computes m-(G, A): the same without LEMMA as the top rule. A lemma's own derivation never needs a lemma at
// its top: if L's derivation starts with lemma M, hoisting M to the outer level costs
// m(M, A) + m(G, A+M) <= m(M, A) + m(L, A+M) + m(G, A+L), and repeating strictly lowers the first lemma's cost. So
// the LEMMA rule here uses m-(L, A) for the lemma and the full m for the continuation; box bodies use the full m.
static int search_reduced(int G, const aset *A, const uint64_t *tA, int b, int lem) {
  if ((++st_calls & 0xFFFF) == 0 && now() > deadline) longjmp(jb, 1);
  if (fast1(G, A) == 1) return 1;
  if (b < 2) return INF;
  int bestc = b + 1;
  curA cur = {NULL, 0};
  aset A2, A3;
  uint64_t t2[TTW_MAX], t3[TTW_MAX];
  int k = kind[G];
  if (k == K_IMP) {
    int a = fl[G], c = fr[G];
    if (a == c) bestc = 2;
    else if (b >= 3) {
      addA(&A2, t2, A, tA, a);
      int r = best(c, &A2, t2, LIM - 2);
      if (r < INF && r + 2 < bestc) bestc = r + 2;
    }
  }
  if (k == K_NOT && b >= 3) {
    addA(&A2, t2, A, tA, fl[G]);
    int r = best(BOTID, &A2, t2, LIM - 2);
    if (r < INF && r + 2 < bestc) bestc = r + 2;
  }
  if (G != BOTID && k != K_NOT && b >= 4) {
    addA(&A2, t2, A, tA, neg(G));
    int r = best(BOTID, &A2, t2, LIM - 3);
    if (r < INF && r + 3 < bestc) bestc = r + 3;
  }
  if (b >= 3)
    for (int j = 0; j < nD; j++) {
      int D = Dl[j];
      if (D == G || !inA(A, D)) continue;
      int a = fl[D], c = fr[D], ca, cb;
      if (a == G) ca = 1;
      else {
        addA(&A3, t3, A, tA, a);
        int ra = best(G, &A3, t3, LIM - 3);
        if (ra >= INF) continue;
        ca = ra + 1;
      }
      if (c == G) cb = 1;
      else {
        addA(&A3, t3, A, tA, c);
        int rb = best(G, &A3, t3, LIM - ca - 1);
        if (rb >= INF) continue;
        cb = rb + 1;
      }
      if (ca + cb + 1 < bestc) bestc = ca + cb + 1;
    }
  if (lem) {
    aset E;   // lemma candidates: entailed by A (others fail semantically), not in A, not G
    entailed_set(&E, tA);
    for (int i = 0; i < NW; i++) E.w[i] &= ~A->w[i];
    E.w[G >> 6] &= ~(1ULL << (G & 63));
    for (int i = 0; i < NW; i++)
      for (uint64_t x = E.w[i]; x; x &= x - 1) {
        int L = i * 64 + __builtin_ctzll(x);
#ifdef NOHOIST
        int r = best_c(L, A, tA, LIM - 1, &cur, 0);
#else
        int r = best_c(L, A, tA, LIM - 1, &cur, 1);
#endif
        if (r >= INF) continue;
        addA(&A2, t2, A, tA, L);
        int r2 = best(G, &A2, t2, LIM - r);
        if (r2 < INF && r + r2 < bestc) bestc = r + r2;
      }
  }
  return bestc <= b ? bestc : INF;
}
#endif

static int search(int G, const aset *A, const uint64_t *tA, int b, choice *ch) {
#if !defined(TOWERS) && !defined(FULLRULES)
  if (!ch) return search_reduced(G, A, tA, b, 1);
#endif
  if ((++st_calls & 0xFFFF) == 0 && now() > deadline) longjmp(jb, 1);
  int bestc = b + 1;
#ifndef TOWERS
  curA cur = {NULL, 0};
#endif
  aset A2, A3;
  uint64_t t2[TTW_MAX], t3[TTW_MAX];
  int k = kind[G];
  // ANDE
  if (G < nf0)
    for (int j = ande_off[G]; j < ande_off[G + 1]; j++) {
      int C = ande_c[j];
      int r = INPA(C, LIM - 1);
      if (r < INF) CONSIDER(r + 1, ande_w[j] == 1 ? R_ANDE1 : R_ANDE2, C);
    }
  // IMPE
  if (G < nf0)
    for (int j = impe_off[G]; j < impe_off[G + 1]; j++) {
      int I = impe_i[j];
      int r = INPA(I, LIM - 1);
      if (r < INF) {
        addA(&A2, t2, A, tA, I);
        int r2 = inp(fl[I], &A2, t2, LIM - 1 - r);
        if (r2 < INF) CONSIDER(r + r2 + 1, R_IMPE, I);
      }
    }
  // DN
#ifndef TOWERS  // DN only from ~~G in S: cost-preserving, see "NO TOWERS" in the header
  if (negid[G] >= 0 && negid[negid[G]] >= 0 && inS[negid[negid[G]]])
#endif
  {
    int nn = neg(neg(G));
    int r = INPA(nn, LIM - 1);
    if (r < INF) CONSIDER(r + 1, R_DN, nn);
  }
  // BOTE
  if (G != BOTID) {
    int r = INPA(BOTID, LIM - 1);
    if (r < INF) CONSIDER(r + 1, R_BOTE, BOTID);
  }
  if (k == K_AND) {
    int r = INPA(fl[G], LIM - 1);
    if (r < INF) {
      addA(&A2, t2, A, tA, fl[G]);
      int r2 = inp(fr[G], &A2, t2, LIM - 1 - r);
      if (r2 < INF) CONSIDER(r + r2 + 1, R_ANDI, 0);
    }
  }
  if (k == K_OR) {
    int r = INPA(fl[G], LIM - 1);
    if (r < INF) CONSIDER(r + 1, R_ORI1, 0);
    r = INPA(fr[G], LIM - 1);
    if (r < INF) CONSIDER(r + 1, R_ORI2, 0);
  }
  if (k == K_IMP && b >= 2) {
    int a = fl[G], c = fr[G];
    if (a == c) CONSIDER(2, R_IMPI, 0);
    else if (b >= 3) {
      addA(&A2, t2, A, tA, a);
      int r = best(c, &A2, t2, LIM - 2);
      if (r < INF) CONSIDER(r + 2, R_IMPI, 0);
    }
  }
  if (k == K_NOT && b >= 3) {
    addA(&A2, t2, A, tA, fl[G]);
    int r = best(BOTID, &A2, t2, LIM - 2);
    if (r < INF) CONSIDER(r + 2, R_NEGI, 0);
  }
  if (G == BOTID) {
    for (int j = 0; j < nS; j++) {
      int X = Sl[j], NX = negid[X];
      if (NX < 0) continue;
      int nxa = inA(A, NX);
      if (!inS[NX] && !nxa) continue;
      if (nxa) {
        int r = INPA(X, LIM - 1);
        if (r < INF) CONSIDER(r + 1, R_NEGE, X);
      } else if (inA(A, X)) {
        int r = INPA(NX, LIM - 1);
        if (r < INF) CONSIDER(r + 1, R_NEGE, X);
      } else {
        int r = INPA(X, LIM - 2);
        if (r < INF) {
          addA(&A2, t2, A, tA, X);
          int r2 = inp(NX, &A2, t2, LIM - 1 - r);
          if (r2 < INF) CONSIDER(r + r2 + 1, R_NEGE, X);
        }
      }
    }
  }
  // classical reductio
  if (G != BOTID && k != K_NOT && b >= 4) {
    addA(&A2, t2, A, tA, neg(G));
    int r = best(BOTID, &A2, t2, LIM - 3);
    if (r < INF) CONSIDER(r + 3, R_RAA, 0);
  }
  // ORE
  if (b >= 3)
    for (int j = 0; j < nD; j++) {
      int D = Dl[j];
      if (D == G) continue;
      int r = INPA(D, LIM - 3);
      if (r >= INF) continue;
      int a = fl[D], c = fr[D], ca, cb;
      addA(&A2, t2, A, tA, D);
      if (a == G) ca = 1;
      else {
        addA(&A3, t3, &A2, t2, a);
        int ra = best(G, &A3, t3, LIM - r - 3);
        if (ra >= INF) continue;
        ca = ra + 1;
      }
      if (c == G) cb = 1;
      else {
        addA(&A3, t3, &A2, t2, c);
        int rb = best(G, &A3, t3, LIM - r - ca - 1);
        if (rb >= INF) continue;
        cb = rb + 1;
      }
      CONSIDER(r + ca + cb + 1, R_ORE, D);
    }
  // lemma
  if (b >= 2)
    for (int j = 0; j < nS; j++) {
      int L = Sl[j];
      if (L == G || inA(A, L)) continue;
      int r = BESTA(L, LIM - 1);
      if (r >= INF) continue;
      addA(&A2, t2, A, tA, L);
      int r2 = best(G, &A2, t2, LIM - r);
      if (r2 < INF) CONSIDER(r + r2, R_LEMMA, L);
    }
  return bestc <= b ? bestc : INF;
}

// ---------- template output (prefix tokens, parsed by certify_fast.py) ----------
static char *ob;
static int olen, ocap;
static void out(const char *fmt, int x) {
  if (olen + 32 >= ocap) longjmp(jb, 4);
  olen += snprintf(ob + olen, ocap - olen, fmt, x);
}
static void emit_best(int G, const aset *A, const uint64_t *tA, int v);
static void emit_inp(int G, const aset *A, const uint64_t *tA, int v) {
  if (inA(A, G)) { out("R0 %d ", G); return; }
  emit_best(G, A, tA, v);
}
static void emit_best(int G, const aset *A, const uint64_t *tA, int v) {
  if (inA(A, G)) { out("R %d ", G); return; }
  choice ch = {R_NONE, 0};
  int c = search(G, A, tA, v, &ch);
  if (c != v) longjmp(jb, 5);
  aset A2, A3;
  uint64_t t2[TTW_MAX], t3[TTW_MAX];
  if (olen + 32 >= ocap) longjmp(jb, 4);
  olen += snprintf(ob + olen, ocap - olen, "%s %d ", RN[ch.rule], G);
  switch (ch.rule) {
    case R_ANDE1: case R_ANDE2: case R_DN: case R_BOTE: emit_inp(ch.p, A, tA, v - 1); break;
    case R_ORI1: emit_inp(fl[G], A, tA, v - 1); break;
    case R_ORI2: emit_inp(fr[G], A, tA, v - 1); break;
    case R_IMPE: {
      int r = inp(ch.p, A, tA, v - 1);
      emit_inp(ch.p, A, tA, r);
      addA(&A2, t2, A, tA, ch.p);
      emit_inp(fl[ch.p], &A2, t2, v - 1 - r);
      break;
    }
    case R_ANDI: {
      int r = inp(fl[G], A, tA, v - 1);
      emit_inp(fl[G], A, tA, r);
      addA(&A2, t2, A, tA, fl[G]);
      emit_inp(fr[G], &A2, t2, v - 1 - r);
      break;
    }
    case R_NEGE: {
      int X = ch.p, NX = negid[X];
      int r = inp(X, A, tA, v - 1);
      emit_inp(X, A, tA, r);
      addA(&A2, t2, A, tA, X);
      emit_inp(NX, &A2, t2, v - 1 - r);
      break;
    }
    case R_IMPI: {
      int a = fl[G], c2 = fr[G];
      out("%d ", a);
      if (a == c2) { out("N ", 0); break; }
      addA(&A2, t2, A, tA, a);
      emit_best(c2, &A2, t2, v - 2);
      break;
    }
    case R_NEGI:
      out("%d ", fl[G]);
      addA(&A2, t2, A, tA, fl[G]);
      emit_best(BOTID, &A2, t2, v - 2);
      break;
    case R_RAA:
      addA(&A2, t2, A, tA, neg(G));
      emit_best(BOTID, &A2, t2, v - 3);
      break;
    case R_ORE: {
      int D = ch.p, a = fl[D], c2 = fr[D];
      int r = inp(D, A, tA, v - 3);
      emit_inp(D, A, tA, r);
      addA(&A2, t2, A, tA, D);
      int ca = 1;
      out("%d ", a);
      if (a == G) out("N ", 0);
      else {
        addA(&A3, t3, &A2, t2, a);
        int ra = best(G, &A3, t3, v - r - 3);
        ca = ra + 1;
        emit_best(G, &A3, t3, ra);
      }
      out("%d ", c2);
      if (c2 == G) out("N ", 0);
      else {
        addA(&A3, t3, &A2, t2, c2);
        emit_best(G, &A3, t3, v - r - ca - 2);
      }
      break;
    }
    case R_LEMMA: {
      int L = ch.p;
      int r = best(L, A, tA, v - 1);
      out("%d ", L);
      emit_best(L, A, tA, r);
      addA(&A2, t2, A, tA, L);
      emit_best(G, &A2, t2, v - r);
      break;
    }
    default: longjmp(jb, 6);
  }
}

// ---------- entry point ----------
// Formulas 0..n-1 given as (kind, left, right); in_s flags S membership (BOT must be in S); atom formulas carry
// their atom index in `left`. Returns: body >= 1 found (minimal, < floor); 0 exhausted (certified >= floor);
// -1 timeout; -2 A-bitset overflow (rebuild with larger NW); -3 formula table overflow; other < 0 internal error.
// stats[0..8] = calls, sem prunes, memo hits, stores, evictions, records (or entries) used, abits, formulas, table
// size; stats[16 + b] = microseconds and stats[80 + b] = records after exhausting body budget b.
int fm_run(int n, const int *k, const int *l, const int *r, const int *in_s, int natoms, int nprem,
           const int *prem, int concl, int floor_body, double secs, double mem_mb, char *outbuf, int outcap,
           int64_t *stats) {
  nf = nf0 = n;
  nabits = 0;
  nS = nD = 0;
  BOTID = -1;
  int rows = 1 << natoms;
  ttw = rows >= 64 ? rows / 64 : 1;
  fullmask_last = rows >= 64 ? ~0ULL : ((1ULL << rows) - 1);
  for (int i = 0; i < n; i++) {
    kind[i] = k[i]; fl[i] = l[i]; fr[i] = r[i]; inS[i] = in_s[i]; negid[i] = -1; abit[i] = -1;
  }
  for (int i = 0; i < n; i++) {  // children always precede parents
    if (kind[i] == K_NOT) negid[fl[i]] = i;
    if (kind[i] == K_BOT) BOTID = i;
    for (int w = 0; w < ttw; w++) {
      uint64_t x;
      switch (kind[i]) {
        case K_ATOM: {
          int a = fl[i];
          x = 0;
          for (int bit = 0; bit < 64 && w * 64 + bit < rows; bit++)
            if (((w * 64 + bit) >> a) & 1) x |= 1ULL << bit;
          break;
        }
        case K_BOT: x = 0; break;
        case K_NOT: x = ~tt[fl[i]][w]; break;
        case K_AND: x = tt[fl[i]][w] & tt[fr[i]][w]; break;
        case K_OR: x = tt[fl[i]][w] | tt[fr[i]][w]; break;
        default: x = ~tt[fl[i]][w] | tt[fr[i]][w]; break;
      }
      if (w == ttw - 1) x &= fullmask_last;
      tt[i][w] = x;
    }
    if (inS[i]) { Sl[nS++] = i; if (kind[i] == K_OR) Dl[nD++] = i; }
  }
  if (BOTID < 0) return -7;
  int na = 0, ni = 0;
  for (int g = 0; g < n; g++) {
    ande_off[g] = na; impe_off[g] = ni;
    for (int j = 0; j < nS; j++) {
      int f = Sl[j];
      if (kind[f] == K_AND && (fl[f] == g || fr[f] == g)) { ande_c[na] = f; ande_w[na] = fl[f] == g ? 1 : 2; na++; }
      if (kind[f] == K_IMP && fr[f] == g) impe_i[ni++] = f;
    }
  }
  ande_off[n] = na; impe_off[n] = ni;
#ifndef TOWERS
  if (n > NW * 64) return -2;
  for (int i = 0; i < n; i++) abit[i] = i;
  nabits = n;
  n_np = 0;
  for (int g = 0; g < n; g++) {
    memset(&m_ande[g], 0, sizeof(aset));
#define SETB(m, f) ((m).w[(f) >> 6] |= 1ULL << ((f) & 63))
    for (int j = ande_off[g]; j < ande_off[g + 1]; j++) SETB(m_ande[g], ande_c[j]);
    SETB(m_ande[g], g);
    if (negid[g] >= 0 && negid[negid[g]] >= 0 && inS[negid[negid[g]]]) SETB(m_ande[g], negid[negid[g]]);
    if (g != BOTID) SETB(m_ande[g], BOTID);
    if (kind[g] == K_OR) { SETB(m_ande[g], fl[g]); SETB(m_ande[g], fr[g]); }
    ip_off[g] = impe_off[g];
  }
  for (int j = 0; j < ni; j++) ip_i[j] = impe_i[j];
  ip_off[n] = ni;
  memset(&all_S, 0, sizeof all_S);
  for (int f = 0; f < n; f++) SETB(all_S, f);
  for (int row = 0; row < rows; row++) {
    memset(&true_at[row], 0, sizeof(aset));
    for (int f = 0; f < n; f++)
      if ((tt[f][row >> 6] >> (row & 63)) & 1) SETB(true_at[row], f);
  }
  for (int j = 0; j < nS; j++) {  // NEGE pairs: X in S with ~X in S (pairs with ~X outside S need ~X in A: impossible)
    int X = Sl[j], NX = negid[X];
    if (NX >= 0 && inS[NX]) { np_x[n_np] = X; np_nx[n_np] = NX; n_np++; }
  }
#endif

#ifndef TOWERS
  if (n > NSLOT) return -9;
#endif
  cap_max = 16;
  while ((cap_max * 2) * sizeof(SLOT_T) <= (uint64_t)(mem_mb * 1048576.0)) cap_max *= 2;
  uint64_t cap = 16;
  while (cap < cap_max && cap * sizeof(SLOT_T) < (1u << 20)) cap *= 2;   // start at ~1 MB, double as needed
  tab = calloc(cap, sizeof(SLOT_T));
  if (!tab) return -8;
  tabmask = cap - 1;
  n_used = n_vals = 0;
  st_calls = st_sem = st_hit = st_store = st_evict = 0;
  ob = outbuf; olen = 0; ocap = outcap; ob[0] = 0;
  deadline = now() + secs;

  int result;
  int code = setjmp(jb);
  if (code) {
    result = code == 1 ? -1 : code == 2 ? -2 : code == 3 ? -3 : -10 - code;
  } else {
    aset A;
    memset(&A, 0, sizeof A);
    uint64_t tA[TTW_MAX];
    for (int i = 0; i < ttw; i++) tA[i] = i == ttw - 1 ? fullmask_last : ~0ULL;
    int concl_in_prem = 0;
    for (int i = 0; i < nprem; i++) {
      if (prem[i] == concl) concl_in_prem = 1;
      aset A2; uint64_t t2[TTW_MAX];
      addA(&A2, t2, &A, tA, prem[i]);
      A = A2; memcpy(tA, t2, sizeof t2);
    }
    result = 0;
    if (!concl_in_prem)
      for (int b = 1; b < floor_body; b++) {   // iterative deepening (measured 2x fewer calls than one search at floor-1)
        int v = best(concl, &A, tA, b);
        if (b < 64) { stats[16 + b] = (int64_t)((now() - deadline + secs) * 1e6); stats[80 + b] = n_used; }
        if (v < INF) {
          result = v;
          ob[0] = 0; olen = 0;
          emit_best(concl, &A, tA, v);
          break;
        }
      }
  }
  stats[0] = st_calls; stats[1] = st_sem; stats[2] = st_hit; stats[3] = st_store; stats[4] = st_evict;
  stats[5] = n_used; stats[6] = nabits; stats[7] = nf; stats[8] = (int64_t)(tabmask + 1); stats[9] = n_vals; stats[10] = sizeof(SLOT_T);
  free(tab);
  tab = NULL;
  return result;
}

// formulas created during the last run (negation towers): id -> (kind, left, right)
int fm_nf(void) { return nf; }
void fm_get(int id, int *k, int *l, int *r) { *k = kind[id]; *l = fl[id]; *r = fr[id]; }
