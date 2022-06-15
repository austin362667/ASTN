# 玩具神經網路 - 以三層為例

- 語言：Go
- 訓練資料集： MNIST 
- 測試資料集：手寫
- 損失函數: MSE
- 矩陣套件: gonum

---
原本打算不動腦直接模仿 Karpathy 的專案 [MicroGrad](https://github.com/karpathy/micrograd/tree/master/micrograd)

但 Python 和 Go 的語法實在差太多了...
最後還是需要仔細思考一下怎麼設計!

(我盡量使用 go 的標準函式庫，除了矩陣預算)

接著是一些數學算式(主要是為了練習 math latex)

#### 執行方式：

`go build` 得到 binary 執行檔

輸入
`./nanograd -mnist train` 以訓練(約5~10 mins)

輸入
`./nanograd -mnist predict` 以預測(約5~10 secs)

---
## 從 input layer 到 hidden layer

$$
\begin{bmatrix} 
w_{11} & w_{21} \\
w_{21} & w_{22} \\
w_{31} & w_{23} \\
\end{bmatrix} 
\cdot
\begin{bmatrix} 
i_1 \\
i_2 \\
\end{bmatrix} 
=
\begin{bmatrix} 
w_{11}*i_1 & w_{21}*i_2 \\
w_{21}*i_1 & w_{22}*i_2 \\
w_{31}*i_1 & w_{23}*i_2 \\
\end{bmatrix} 
$$

---
## 用線性回歸理解損失函數(的偏微分)

### MSE
$$
Error_{(m, b)}
=
\frac{1}{N}
\sum_{i=1}^N (y_i - (m*x_i+b))^2
$$

### Partial Derivative

 - partial `m`
$$
\frac{\partial}{\partial m}
=
\frac{2}{N}
\sum_{i=1}^N -x_i*(y_i - (m*x_i+b))
$$

 - partial `b`
$$
\frac{\partial}{\partial b}
=
\frac{2}{N}
\sum_{i=1}^N -(y_i - (m*x_i+b))
$$

---
## 從 hidden layer 到 output layer

$$
output_k
=
sigmoid(sum(\begin{bmatrix} 
w_{11}*o_{k1} \\
w_{21}*o_{k2} \\
w_{31}*o_{k3} \\
\end{bmatrix}))
$$

---
## 我們的損失函數(與梯度)

$$
E_k = target_k - output_k
$$

求梯度
$$
\nabla w_{jk}
=
\frac{\partial E_k}{\partial o_k}
*
\frac{\partial o_k}{\partial \sum_k}
*
\frac{\partial \sum_k}{\partial w_{jk}}
$$

經過

$$
\frac{\partial E_k}{\partial o_k}
=
- (t_k - o_k)
$$

$$
\frac{\partial o_k}{\partial \sum_k}
=
o_k(1-o_k)
$$

$$
\frac{\partial \sum_k}{\partial w_{jk}}
=
o_j
$$

得

$$
\nabla w_{jk}
=
- (t_k - o_k)
*
o_k(1-o_k)
*
o_j
$$

---

## 反傳遞

根據前一個隱藏層的連接，輸出層的誤差是由來自隱藏層的誤差貢獻的。

換句話說，隱藏層的誤差組合形成了輸出層的誤差。

記得要 `Transpose` 權重矩陣

$$
\begin{bmatrix} 
w_{11} & w_{21} \\
w_{21} & w_{22} \\
w_{31} & w_{23} \\
\end{bmatrix} 
\cdot
\begin{bmatrix} 
e_1^{output} \\
e_2^{output} \\
\end{bmatrix} 
=
\begin{bmatrix} 
w_{11}*e_1^{output} & w_{21}*e_2^{output} \\
w_{21}*e_1^{output} & w_{22}*e_2^{output} \\
w_{31}*e_1^{output} & w_{23}*e_2^{output} \\
\end{bmatrix}
=
\begin{bmatrix} 
e_1^{hidden} \\
e_2^{hidden} \\
e_3^{hidden} \\
\end{bmatrix} 
$$

---

### 學習率

$$
\nabla w_{jk}
=
- lr * (t_k - o_k)
*
o_k(1-o_k)
*
o_j
$$

最後乘上學習率`learning rate (lr)`就完成啦！恭喜～

---

### 參考資料

- [梯度下降算法 by 陳鍾誠老師](https://gitlab.com/ccc110/ai/-/tree/master/07-neural/02-gradient)
- [Cross Entropy vs. MSE as Cost Function for Logistic Regression for Classification](https://www.youtube.com/watch?v=m0ZeT1EWjjI)
- [An Introduction to Gradient Descent and Linear Regression](https://spin.atomicobject.com/2014/06/24/gradient-descent-linear-regression/)
