# TouchDesigner の送信用プロジェクト（sender）

`td/transform_sender.toe` は、左右の腕の目標を受け取り、範囲の制限と安全確認をしてから、bridge へ OSC で送ります。目標は、プロジェクトの中のスライダーか、ほかの TouchDesigner のプロジェクトから受け取ります。

TouchDesigner 2025.33230 で作っています。

## 1. 画面のスイッチ

| スイッチ | 場所 | 働き |
|---|---|---|
| 送り先（MuJoCo／Real robot） | `/project1/container_target` | OSC の宛先を選ぶ。MuJoCo は 127.0.0.1。Real robot は `/project1/table_addr` の `Real robot` の行のアドレス（Ubuntu の PC のアドレスに書き換える） |
| Input source（Slider／TouchIn） | `/project1/container_input/container_input_source` | 目標をスライダーから取るか、ほかのプロジェクトから受け取るかを選ぶ（2 節） |
| Right arm enable・Left arm enable | `/project1/container_real_robot_enable` | 実機の腕ごとの Enable（MuJoCo では使わない） |
| Move HOME | `/project1/container_robot_control` | 押すと Home へ移動する |
| Enable sending data | `/project1/container_robot_control` | Follow の主スイッチ。OFF にすると必ず止まる |
| Open logger・Clear logger | `/project1/container_logger_control` | bridge からの状態の表を開く・消す（5 節） |
| Open final state monitor | `/project1/container_logger_control` | 最後に送る左右の目標と違反の状態を表示する（5 節） |


送るのは毎フレーム（240 fps）です。Follow を OFF にしても、目標と各スイッチの値は送り続けます（Follow は bridge の動作の切り替えで、送信そのものの ON／OFF ではありません）。

## 2. 目標の入力

Input source で、両腕まとめて切り替えます（`/project1/container_input/switch_source` の Index が切り替わります）。

| Input source | 入力 |
|---|---|
| Slider | スライダー（`/project1/container_input/container_gui`） |
| TouchIn | ほかのプロジェクトからの入力（Touch In CHOP `/project1/container_input/touchin_openarm`、Port 9310。形は [osc-protocol.md](osc-protocol.md) の「sender へ送る形」） |

### スライダー

腕ごと（`container_transform_right_gui`・`container_transform_left_gui`）に、次のスライダーがあります。

| スライダー | 送る値 |
|---|---|
| px, py, pz | Home の手先からの位置（各 ±0.2 m） |
| rx, ry, rz | Home の手先からの回転（各 ±18°。クォータニオンにして送る） |
| gripper | 0〜1 を 0〜−1 にして送る（bridge が −0.785〜0 に切り詰める） |

位置と回転は、このあと 3 節の範囲で制限されます。

座標は `arm_origin`（x 前・y 左・z 上）で、Home の手先の位置からの相対です。回転は、Home の手先の回転に右から掛けます（手先の座標での回転）。

## 3. 範囲の制限と安全確認

場所：`/project1/container_input/safety`

| 処理 | 内容 |
|---|---|
| 位置の範囲 | 腕・方向ごとに制限する（下の表）。超えるとその位置で止まり、戻せばまたついてくる |
| 回転の範囲 | 回転を手先の軸のオイラー角（度）にし、各軸 −25〜25° に制限してから戻す（Constant `rot_min`・`rot_max`） |
| 値の急な変化 | 制限後の目標が 0.1 秒で 0.10 m 以上、または 45° 以上変わったら違反。入力の `r_active`・`l_active` と、それを 0.2 秒遅らせた値がどちらも 1 のときだけ見る |
| 入力の途絶 | 入力の `heartbeat` が 0.1 秒変わらなければ、両腕とも違反 |

位置の範囲（Home からの相対、m。Constant `range_min`・`range_max`）：

| 方向 | 右腕（値の場所） | 左腕（値の場所） |
|---|---|---|
| 前 | 0.17（`range_max` `r_px`） | 0.17（`range_max` `l_px`） |
| 後ろ | 0.025（`range_min` `r_px` = −0.025） | 0.025（`range_min` `l_px` = −0.025） |
| 内側 | 0.14（`range_max` `r_py`） | 0.14（`range_min` `l_py` = −0.14） |
| 外側 | 0.17（`range_min` `r_py` = −0.17） | 0.17（`range_max` `l_py`） |
| 上 | 0.17（`range_max` `r_pz`） | 0.17（`range_max` `l_pz`） |
| 下 | 0.15（`range_min` `r_pz` = −0.15） | 0.15（`range_min` `l_pz` = −0.15） |

- y は左が＋なので、内側は右腕では＋、左腕では−の向きです
- bridge は、位置の大きさが 0.30 m（`config/control.yml` の `max_move_m`）を超えた値を捨て、0.2 秒届かないとその腕を止めます。範囲を決めるときは、角の大きさ √(max(前,後ろ)² ＋ max(外側,内側)² ＋ max(上,下)²) を 0.30 m 未満にします（上の値で 0.294 m）。どれか 1 つを広げるときは、ほかを狭めます。
- 前・内側・下は、Home の手の向きのまま腕が届く距離（約 0.17・0.14〜0.15・0.15 m）に合わせています。それより広げても腕は動きません。
- 後ろは、後ろ・内側・上へ同時に動かしたときに支柱へ近づくので、狭くしています。
- **支柱やもう一方の腕との衝突は、sender・bridge・IK のどこでも防いでいません。** 両手を中央で近づけると、腕どうしが当たることがあります。操作する人が見て避けてください。
- `/project1/container_input/safety/switch_safety_limit` を 1 にすると、左右の内側の量の合計を制限する処理（`collision_safety`）に切り替わります。合計が上限を超えると、両手が外側へ押し戻されます。

## 4. Follow の許可

場所：`/project1/follow_gate`

bridge へ送る Follow ＝ Enable sending data × （1 − 次のどれか）

- Enable を ON にした腕の違反（ラッチして、消えるまで止めたままにする）
- 今の違反
- Enable を ON にした腕の `active` が 0

ラッチは、Enable sending data を OFF にしたとき、または Enable を ON にした腕の `active` がすべて 0 になったときに消えます。

- Enable が OFF の腕の違反は見ません。MuJoCo では Enable を使わずに両腕が動くので、違反で止めたいときは、使う腕の Enable を ON にしてください。
- スライダーのとき、`active` は 1、`heartbeat` は sender の中で作った値です。
- bridge は、Home の完了後に Follow が 0 から 1 に変わったときだけ追従を始めます。Follow が 0 になると、再開には Home が要ります。

## 5. 状態の表示

Open logger で、bridge からの状態の表（`/project1/container_logger`、UDP 7001 で受信）を開きます。Clear logger で表を消します。

- `status_now`：左右の今の状態（1 腕 1 行、通知のたびに書き換える）
- `status_log`：状態や理由が変わったときだけ 1 行足す履歴

主な状態：

| 表示 | 意味 |
|---|---|
| `Stopped` | 止まっている（Enable OFF） |
| `Starting` | 実機の開始の動作中 |
| `Waiting for joint state` | 開始の後、関節の状態を待っている |
| `Waiting for Home` | Move HOME を受け付ける |
| `Homing` | Home へ移動中 |
| `Home done (Follow allowed)` | Enable sending data の OFF→ON を受け付ける |
| `Following` | 追従中 |
| `Stop requested`・`Returning (normal stop)`・`Confirming disable` | 通常停止の途中 |
| `FAULT (check robot)` | 異常。モーターが切れたことを確かめられていない |

同じ UDP 7001 を使うため、`status_monitor.py`（端末で状態を表示するモニター）と同時には使えません。

Open final state monitor で開く表示（`/project1/container_final_state`）には、範囲の制限の後に bridge へ送る左右の目標（`/project1/transform_left`・`transform_right`）と、違反の状態（`/project1/violation`）が出ます。

## 6. 送信

`/project1/execute3` が毎フレーム、`/project1/transform_left`・`transform_right` の最後のサンプル（8 値）、Home、Follow、左右の Enable を `/project1/oscout_dat` から送ります（アドレスは `/project1/table_osc_address`）。形は [osc-protocol.md](osc-protocol.md)。
