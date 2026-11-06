# 通信の形式

## 1. bridge が受け取る OSC（UDP 7000）

ポートは `config/control.yml` の `osc_port` で変えられます。bridge は MuJoCo では 127.0.0.1、実機では 0.0.0.0 で受けます（`dataflow-*.yml` の `--listen`）。

| アドレス | 値 | 意味 |
|---|---|---|
| `/openarm/right` | `[px, py, pz, qw, qx, qy, qz, gripper]`（8 個の float） | 右腕の目標 |
| `/openarm/left` | 同上 | 左腕の目標 |
| `/openarm/home` | `0` か `1` | 0→1 で Home へ移動 |
| `/openarm/follow` | `0` か `1` | Home の完了後の 0→1 で追従を始め、1→0 で止める |
| `/openarm/enable_right` | `0` か `1` | 実機の右腕の開始（1）・通常停止（0）。MuJoCo では使わない |
| `/openarm/enable_left` | `0` か `1` | 実機の左腕の開始・通常停止 |

目標の値：

| 値 | 内容 |
|---|---|
| `px, py, pz` | Home の手先の位置からの相対の位置（m）。座標はモデルの `arm_origin`（x 前・y 左・z 上） |
| `qw, qx, qy, qz` | Home の手先の回転からの相対の回転（単位クォータニオン）。Home の回転に右から掛ける（手先の座標での回転） |
| `gripper` | −π/4（約 −0.785）〜0。左右とも同じ向きで送り、左は bridge が符号を反転する |

bridge が受け付けない値（ログに `[OSC] Rejected ...` が出て、その値は使わない）：

- 8 個でない、または有限でない値
- 位置の大きさ $\sqrt{\mathrm{px}^{2} ＋ \mathrm{py}^{2} ＋ \mathrm{pz}^{2}}$ が `max_move_m`（0.30 m）を超える値
- クォータニオンの長さが 0 の値（長さはそれ以外なら 1 にそろえる）
- スイッチが 0 か 1 以外の値

受け付けない値は「届いていない」と同じ扱いです。0.2 秒（`timeout_s`）新しい目標が届かないと、その腕を止めます。

- 目標と各スイッチは、送り続けてください（sender は 240 fps で送ります）。bridge は 100 Hz で最新の値を使います。
- Home と Follow は、0→1 に変わったときだけ受け付けます。押しっぱなしでは繰り返しません。

## 2. sender へ送る形（Touch In CHOP、Port 9310）

ほかの TouchDesigner のプロジェクトから `td/transform_sender.toe` に目標を送るときは、Touch Out CHOP で次の 19 チャンネルを送り、sender の Input source を TouchIn にします（[td-sender.md](td-sender.md)）。

| チャンネル | 内容 |
|---|---|
| `r_px r_py r_pz` | 右腕の位置（1 節の `px, py, pz` と同じ） |
| `r_qw r_qx r_qy r_qz` | 右腕の回転（1 節の `qw, qx, qy, qz` と同じ） |
| `r_gripper` | 右のグリッパー（−0.785〜0） |
| `l_px` 〜 `l_gripper` | 左腕の同じ 8 個 |
| `r_active l_active` | 1 のときだけ、その腕を動かしてよい。0 にすると sender は Follow を切る（再開には Home が要る） |
| `heartbeat` | 送り手が動いている間、変わり続ける値（例：LFO CHOP の Ramp）。0.1 秒変わらないと、sender は両腕を止める |

- 動かさない腕は、位置 0、回転 (1, 0, 0, 0) を送ります。
- 原点（Home の手先に当たる手の位置）を取り直したときに目標が 0 へ戻るのは、`r_active`・`l_active` を 0 から 1 にした直後なら止まりません（sender は active が 1 になってから 0.2 秒の間、急な変化を見ないため）。

## 3. 状態の通知（UDP 7001）

bridge は、状態や理由が変わったときすぐに、変わらなくても 0.5 秒ごとに、腕ごとの状態を `config/control.yml` の `status.host`・`status.port` へ送ります。表示のための通知で、ここからロボットを操作することはありません。

アドレスは `/openarm/status/right` と `/openarm/status/left`。値は次の 11 個です。

| 番号 | 内容 |
|---|---|
| 0 | 起動ごとのセッションの文字列 |
| 1 | 通し番号 |
| 2 | 状態（`STOPPED`、`STARTING`、`WAIT_STATE`、`WAIT_HOME`、`HOME`、`READY`、`FOLLOW`、`STOP_REQUESTED`、`STOPPING`、`DISABLING`、`FAULT`） |
| 3 | 理由の文字列 |
| 4・5 | 実測の手先と Home の位置の差（m）・姿勢の差（度）。分からないときは −1 |
| 6・7 | Move HOME を受け付けるか・Follow を受け付けるか（0 か 1） |
| 8 | モーターの無効化の応答（1＝確認済み、0＝未確認、−1＝まだない） |
| 9・10 | 重力補償の有無（0 か 1）・補正の割合 |

- 受け手は、古いセッションや古い通し番号の値を捨ててください
- 2 秒新しい通知が届かなければ、状態は分からない（表示を信じない）と扱ってください
- 「無効化の応答：確認済み」はソフトウェアでの応答の確認で、腕が機械的に止まって支えられていることを保証するものではありません
