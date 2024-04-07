# Digital-control (Yasundo Takahashi)
“高橋安人，ディジタル制御，岩波書店， 1987 ，第4刷”


本書のプログラムは行列演算を行えるHP-Basicで記述されています。
今回、多くの方が使い易いようにPythonで再記述してみました。

本書の面白いところは、全てのアルゴリズムが、HP-Basi言語と付属の基本行列演算で作られている点です。正準形変換の方法やリカチ行列の解法などの記述が多々あり大変興味深いものです。これらは現在ではpython-controlなどを使えば簡単に解けます。しかし、その解法の基本原理まで踏み込み、意外と簡単なアルゴリズムで計算できることを本書を通じて実感できることは楽しいです。なお本解説は本人の再学習のためであり、主な内容のメモとプログラムの記述だけです。デジタル制御の解説ではありません
############################################################################

DGC_no1 page1-53　
１．緒言、２．ｚ変換と反復計算、３．線形離散時間系
  
DGC_no2 page55-78　
４.連続時間プラントの離散時間形　4.1 プラント微分方程式と等価の差分式　4.2 パルス伝達関数と拡張ｚ変換　4.3 解析的手法によるｚ変換と拡張ｚ変換　4.4 時系列からの伝達関数 4.5　観測器による状態推定
(20231227 add shimojo)

  
DGC_no3 page79-95　
５.単一ループディジタル制御　5.1　フィードバック制御系　5.2　１次プラントの制御　5.3　追従系の設計　5.4　PID制御則　5.5　PIDゲイン調整


DGC_no4 page97-120　
6.　線形2乗最適制御 6.1　行列リカチ(Riccati)式 6.2　慣性体のLQ制御 6.3　I動作を含むLQ制御 6.4　 観測器を含むLQI制御 6.5　０型プラントのLQI制御


DGC_no5 page121-146　
7.　固有値を指定した制御系　7.1　LQ制御系の固有値　7.2　LQ制御系の根軌跡　7.3　固有値を指定の制御系および状態推定系設計　7.4　有限整定系　7.5　多変数制御系







