# include <stdio.h>
# include <string.h>
# include<stdlib.h>
# include <math.h>
# include <malloc.h>

const
float
alpha = 0.5; // 目标函数中的超参数
const
float
gamma = 0.1; // 目标函数中的超参数

float ** couples; // 存放因果对向量
long * order_neg; // 生成假因果对，序号表示因，取值表示果，对应couples中因果对的序号
float * t; // 需要优化的参数
float * tao; // 需要优化的参数
long
count, size; // 因果对的数量；向量维度

float
eta; // 优化中学习步长
float ** diff, ** diff_neg; // P和P - 中因果向量之差，c - e, c'-e'

void
shuffling(long * arr, long
leng) {
    time_t
tt;
int
r, i;
long
temp;

srand((unsigned)
time( & tt));
for (i = leng-1; i > 0; i--) {
    r = (int)(rand() / (RAND_MAX + 1.0) * i);
// swap(arr[i], arr[r]);
temp = arr[i];
arr[i] = arr[r];
arr[r] = temp;
}
}

void
normalvector(float * a, int
num) {
    int
i;
float
sum;
sum = 0;
for (i = 0; i < num; i++)
{
    sum += a[i] * a[i];
}
sum = sqrt(sum);
for (i = 0; i < num; i++)
{
    a[i] = a[i] / sum;
}
}

float
norm(float * a, int
num) {
    float
sum;
int
i;

sum = 0;
for (i = 0; i < num; i++)
{
    sum += a[i] * a[i];
}
sum = sqrt(sum);

return sum;
}

void
train()
{
int
j, k, Tmax;
int
id;
float
sum;
float * gt, *gtao;
float
eps;
float
f, *f_ce1, *f_ce2, *f_ce_neg1, *f_ce_neg2;

eps = 1e-3;
Tmax = 100000;
eta = 0.5;

gt = (float *)
malloc(size * sizeof(float));
gtao = (float *)
malloc(size * sizeof(float));
f_ce1 = (float *)
malloc(size * sizeof(float));
f_ce2 = (float *)
malloc(size * sizeof(float));
f_ce_neg1 = (float *)
malloc(size * sizeof(float));
f_ce_neg2 = (float *)
malloc(size * sizeof(float));
for (k = 0; k < Tmax; k++) {
                           // 正则项求梯度
sum = 0;
for (j = 0; j < size; j++) {
gt[j] = t[j] + tao[j];
sum += gt[j] * gt[j];
}
sum = sum * sqrt(sum);
for (j = 0; j < size; j++) {
gt[j] = -gt[j] * alpha / sum;
gtao[j] = gt[j];
}

// 损失项求梯度
id = (int)(rand() / (RAND_MAX + 1.0) * count); // 随机选择因事件 c
f = gamma;
for (j = 0; j < size; j++) {
f_ce1[j] = diff[id][j] + t[j];
f_ce2[j] = tao[j] - diff[id][j];
f_ce_neg1[j] = diff_neg[id][j] + t[j];
f_ce_neg2[j] = tao[j] - diff_neg[id][j];

f += fabsf(f_ce1[j]) + fabsf(f_ce2[j]) - fabsf(f_ce_neg1[j]) - fabsf(f_ce_neg2[j]);
}

if (f > 0) {
for (j = 0; j < size; j++) {
if (f_ce1[j] > 0) gt[j] += 1;
if (f_ce1[j] < 0) gt[j] -= 1;
if (f_ce_neg1[j] > 0) gt[j] -= 1;
if (f_ce_neg1[j] < 0) gt[j] += 1;

if (f_ce2[j] > 0) gtao[j] += 1;
if (f_ce2[j] < 0) gtao[j] -= 1;
if (f_ce_neg2[j] > 0) gtao[j] -= 1;
if (f_ce_neg2[j] < 0) gtao[j] += 1;
}
}

// 更新参数 t, tao
if (k > 100) eta = eta / k;
for (j = 0; j < size; j++) {
gt[j] *= eta;
gtao[j] *= eta;

t[j] -= gt[j];
tao[j] -= gtao[j];
}

// 迭代结束判断
if (norm(gt, size) / norm(t, size) < eps & & norm(gtao, size) / norm(tao, size) < eps)
break;
}

free(gt);
free(gtao);
free(f_ce1);
free(f_ce2);
free(f_ce_neg1);
free(f_ce_neg2);
}

int
main(int
argc, char ** argv) {
    FILE * f;
char
file_name[200];
long
i, j;

if (argc < 2)
{
printf("Usage: ./event-causality <FILE>\nwhere FILE contains event projections\n");
return 0;
}
strcpy(file_name, argv[1]);
f = fopen(file_name, "r");
if (f == NULL)
{
printf("Input file not found\n");
return -1;
}

fscanf(f, "%ld, %ld", & count, & size);
couples = (float **)
malloc(count * sizeof(float *));
for (i = 0; i < count; i++)
{
couples[i] = (float *)
malloc(size * sizeof(float));
for (j = 0; j < size; j++) {
    fscanf(f, "%f,", & couples[i][j]);
}
normalvector(couples[i], size);
i + +;
couples[i] = (float *)
malloc(size * sizeof(float));
for (j = 0; j < size; j++) {
    fscanf(f, "%f,", & couples[i][j]);
}
normalvector(couples[i], size);
}
fclose(f);

count = count / 2;
order_neg = (long *)
malloc(count * sizeof(long));
for (i = 0; i < count; i++)
{
order_neg[i] = i;
}
shuffling(order_neg, count);

diff = (float **)
malloc(count * sizeof(float *));
diff_neg = (float **)
malloc(count * sizeof(float *));
for (i = 0; i < count; i++)
{
diff[i] = (float *)
malloc(size * sizeof(float));
for (j = 0; j < size; j++) {
    diff[i][j] = couples[i * 2][j] - couples[i * 2 + 1][j];
}

diff_neg[i] = (float *)
malloc(size * sizeof(float));
for (j = 0; j < size; j++) {
    diff_neg[i][j] = couples[i * 2][j] - couples[2 * order_neg[i] + 1][j];
}
}

t = (float *)
malloc(size * sizeof(float));
tao = (float *)
malloc(size * sizeof(float));
for (i = 0; i < size; i++)
{
t[i] = (rand() / (float)RAND_MAX - 0.5) / size;
tao[i] = (rand() / (float)RAND_MAX - 0.5) / size;
}

train();

f = fopen("result", "w");
fprintf(f, "%d\n", size);
for (j = 0; j < size-1; j++)
{
fprintf(f, "%f, ", t[j]);
}
fprintf(f, "%f\n", t[size - 1]);
for (j = 0; j < size-1; j++)
{
fprintf(f, "%f, ", tao[j]);

}
fprintf(f, "%f\n", tao[size - 1]);
fclose(f);

for (i = 0; i < count; i++)
{
free(diff[i]);
free(diff_neg[i]);
free(couples[2 * i]);
free(couples[2 * i + 1]);
}
free(couples);
free(diff);
free(diff_neg);
free(order_neg);
free(t);
free(tao);

return 0;
}
