# Hardware setup

TouchDesigner を動かす PC（以下「TD の PC」）から、LAN で Ubuntu PC へ OSC を送り、Ubuntu の PC の bridge・IK・実機のノードで OpenArm v2 Pedestal を動かします。

**非常時は、実機の非常停止を使ってください。** ソフトウェアの停止は、非常停止の代わりになりません。初めて動かすときは、腕を支えられる状態にし、開始・帰還の動作の経路に人や物がないことを確かめてください。

## 1. 動きの概要

| 操作 | 実機の動き |
|---|---|
| Enable を ON | ドライバーを開始し、開始の姿勢へ 10 秒かけて移動する |
| Move HOME | Home の姿勢へ 4 秒かけて移動する（手先の姿勢を補間） |
| Enable sending data を ON | Home の完了後、目標に追従する（Follow） |
| Enable sending data を OFF | 最後の指令の位置で止まる。再開には Move HOME が要る |
| Enable を OFF | 通常停止：終了の姿勢へ 10 秒かけて戻り、グリッパーを閉じ、モーターを無効にする |

通常停止は、帰還の動作とグリッパーを閉じる動作を伴います（すぐにモーターを切るわけではありません）。

## 2. Ubuntu PC の準備

### 環境を作る

```bash
git clone <このリポジトリ>
cd openarm-controller-touchdesigner
uv sync --extra sim --extra hardware
```

`hardware` の依存（実機のドライバー）は Linux だけで入ります。

### CAN

右腕を `can0`、左腕を `can1` につなぎます（`config/openarm_pedestal.yaml` で変えられます）。

```bash
ip -details link show can0
ip -details link show can1
```

どちらも `UP` で、`bitrate 1000000`・`dbitrate 5000000` であることを確かめます。そうなっていないとき（PC を再起動したときなど）は、OpenArm CAN CLI tool `openarm-can-cli` で設定します。`openarm-can-cli` の入れ方と使い方は、[OpenArm の公式の説明](https://docs.openarm.dev/api-reference/can/cli/)を見てください。

```bash
openarm-can-cli -i can0 can_configure -b 1000000 -d 5000000 --fd --sp 0.75 --dsp 0.75 --dsjw 2 --rm 0
openarm-can-cli -i can1 can_configure -b 1000000 -d 5000000 --fd --sp 0.75 --dsp 0.75 --dsjw 2 --rm 0
```

モーターがすべて応答するかを確かめます（読み取り）。

```bash
openarm-can-cli -i can0 discover
openarm-can-cli -i can1 discover
```

左右とも **8 個**（ID 1〜8）見つかれば準備ができています。8 個未満のときは先に進まず、ケーブルの接続とモーターの電源を見直してください。

- `change_id`・`change_baud`・`write_param`・`set_zero` はモーターの中の設定を書き換えます

## 3. ネットワークの設定

TD の PC と Ubuntu の PC を同じ LAN につなぎ、それぞれの IPv4 アドレスを決めておきます。

| 設定する場所 | 値 |
|---|---|
| `td/transform_sender.toe` の送り先（`/project1/table_addr` の `Real robot` の行） | Ubuntu PC のアドレス |
| Ubuntu PC の `config/control.yml` の `status.host` | TD の PC のアドレス（状態の通知の宛先） |

- bridge は UDP 7000 で受け、UDP 7001 へ状態を送ります。ファイアウォールで止められている場合は、Ubuntu PC の UDP 7000 の受信と、TD の PC の UDP 7001 の受信を許可してください
- Windows で UDP 7001 の受信を許可する例（管理者の PowerShell。`<Ubuntu の PC のアドレス>` を置き換える）：

```powershell
New-NetFirewallRule -DisplayName "OpenArm status UDP 7001" -Direction Inbound -Action Allow -Protocol UDP -LocalPort 7001 -RemoteAddress <Ubuntu の PC のアドレス>
```

## 4. 実機用の設定

`config/control.yml` で次を決めます。起動時に読むので、変えたら Dora を起動し直します。

```yaml
hardware_gravity_compensation:
  enabled: true
  scale: 1.0
  max_offset_rad: 0.15
home_diagnostics: true
```

- 重力補償は、`scale` × モデルの重力トルク ÷ 実機の Kp を関節の目標に足し、各関節 ±0.15 rad に制限します。グリッパーには足しません。`scale` は 0〜1 です。
- 実機は、モデルより重力によるたわみが大きいことがあります。補償がないと、Home の許容範囲（位置 2 cm・姿勢 8°）に入らずに Home が完了しないことがあります。
- `home_diagnostics: true` にすると、Home の誤差の詳しい JSON をログに出します。

主な値（`config/control.yml`・`config/openarm_pedestal.yaml`）：

| 項目 | 値 |
|---|---|
| Kp | 70, 70, 70, 60, 10, 10, 10, 10 |
| Kd | 2.75, 2.5, 2, 2, 0.7, 0.6, 0.5, 0.2 |
| Home の関節の速さの上限 | 0.5 rad/s |
| Follow の関節の速さの上限 | 0.75 rad/s（すべての関節を同じ比率で遅くする） |
| Follow の手先の位置の速さの上限 | 0.2 m/s（IK の前に制限） |
| グリッパーの速さの上限 | 2.0 rad/s（つかむ力の上限はドライバーの設定で 1 Nm） |
| Home の補間／タイムアウト | 4 秒／15 秒 |
| Home の完了の条件 | 実測の手先が位置 2 cm・姿勢 8° 以内 |
| OSC・関節の観測のタイムアウト | 0.2 秒 |
| 開始・通常停止の移動 | 10 秒・50 Hz |

これらは動作確認に使っている値で、安全を保証する値ではありません。

## 5. 起動

Ubuntu PC で、リポジトリのルートから実行します。

```bash
bash run-hardware.sh
```

ログに次の行が出ることを確かめます。

```text
[GRAVITY] hardware enabled=True scale=1 ...
[CONTROL] Home=0.5 Follow=0.75 rad/s, hand=0.2 m/s; ...
[RECORD] off
```

Follow の間の記録を取るときは `bash run-hardware.sh --record` で起動します（7 節）。

## 6. 操作

TD の PC で `td/transform_sender.toe` を開き、送り先を **Real robot** にします。左右の Enable・Move HOME・Enable sending data は、すべて OFF にしておきます。状態は sender の状態の表（[td-sender.md](td-sender.md) の「状態の表示」）で見ます。

| 順 | 操作 | 待つ・確かめること |
|---|---|---|
| 1 | 使う腕の Enable を ON | `Starting` → `Waiting for Home` |
| 2 | Move HOME | `Homing` → `Home done (Follow allowed)` |
| 3 | Enable sending data を ON | `Following`。最初はスライダーを少しずつ動かす |
| 4 | Enable sending data を OFF | 止まる。続けるときは 2 から |
| 5 | Enable を OFF | `Returning (normal stop)` → `Confirming disable` → `Stopped`（無効化の応答が confirmed） |
| 6 | Ubuntu の PC の端末で `Ctrl+C` | Dora が終わる |

- Home の完了前の Follow や、停止の途中の Enable は、理由を返して拒否します。後から勝手に始まりません。拒否されたら、そのスイッチを OFF→ON し直します。
- 次のときは、その腕を自動で止めます（通常停止）。通信が戻っても自動では再開しません。再開には Enable を OFF→ON にし、Home から行います。
  - TouchDesigner からの値が 0.2 秒届かない
  - Home や Follow の間に、関節の観測が 0.2 秒届かない
  - Home が 15 秒で完了しない
  - Home を始めたときの関節の指令と実測の差が 0.1 rad を超えた（`startup tracking error exceeds driver alignment threshold`）
  - ドライバーが安全のために指令を拒否した
- 無効化の送信に失敗した、または無効化の応答を確かめられないときは `FAULT (check robot)` になります。モーターが切れたとは扱いません。実機を確かめてください。

## 7. 記録

`--record` を付けて起動すると、bridge が Follow の間だけ、10 ms ごと（制御の周期ごと）に記録します。付けなければ記録しません。書き出し先は `config/control.yml` の `record_dir` です（相対パスはリポジトリのルートから）。

- 起動ごとに `<record_dir>/<起動の日時>/` を作ります。
- Follow を始めるたびに区間の番号を 1 つ増やし、腕ごとに `follow_<番号>_<right|left>.csv` を書きます。Follow 以外の間は書きません。
- `events.csv`：Follow の始まりと終わり（`wall_ns, t_s, side, event, segment, reason`）。終わりの理由（Follow OFF、OSC timeout など）が残ります。
- 書き込みはためておき、約 1 秒ごとと区間の終わりにまとめて書きます。書けないときは、エラーで bridge を止めます（腕は終了の処理で止まります）。

`follow_*.csv` の列（1 行 ＝ 制御の 1 周期）：

| 列 | 内容 |
|---|---|
| `t_s` | Follow を始めてからの秒（`time.perf_counter()`） |
| `wall_ns` | その PC の時刻（ns）。ほかの PC の記録と合わせるときに使う |
| `osc_age_s`、`osc_px`〜`osc_gripper` | 受け取った OSC の目標（8 値）と、受け取ってからの秒 |
| `target_px`〜`target_qz`、`target_gripper` | 手先の速さを制限した後の、IK へ送った目標（`arm_origin` の座標、m・wxyz）とグリッパー |
| `ik_age_s`、`ik_q1`〜`ik_gripper` | 最新の IK の答えと、その古さ |
| `command_age_s`、`command_q1`〜`command_gripper` | 重力補償と速度の制限をかけて送った指令と、その古さ |
| `gravity_q1`〜`gravity_gripper` | 指令に足した重力補償 |
| `state_age_s`、`observation_ns`、`state_q1`〜`state_gripper` | 実測の関節と、その古さ・実機での観測の時刻（ns） |
| `driver_command_q1`〜`driver_command_gripper` | ドライバーが最後に受け付けた指令（実機だけ。MuJoCo では空） |

- まだ届いていない値は空欄です。
- 手先の実測の姿勢は記録しません。解析では、`state_*` から公式のモデルの FK で求めます。
- `*_age_s` は制御が使う時計（`time.monotonic()`）で測ります。Windows では 15.6 ms 刻みになります。

## 8. うまくいかないとき

| 症状 | 確かめること |
|---|---|
| `discover` で 8 個未満 | ケーブルの接続、モーターの電源 |
| 状態の表が更新されない（`NO STATUS`） | Dora が動いているか、`config/control.yml` の `status.host`、TD の PC のファイアウォール（UDP 7001）、`status_monitor.py` が同じ 7001 を使っていないか |
| `Home timed out` | 状態の表の Home の誤差と、ログの Home の JSON。重力補償の設定 |
| `startup tracking error exceeds driver alignment threshold` | Home を始めたときの関節のずれが大きい。一度 Enable を OFF にしてやり直す |
| Enable や Move HOME が拒否される | 状態の表の理由を読み、そのスイッチを OFF→ON し直す |
| Enable sending data を ON にしても動かない | `Home done (Follow allowed)` になってから ON にしたか。sender の Input source と Enable の状態（[td-sender.md](td-sender.md)） |
| ログに `[OSC] Rejected ...` が出続ける | 送っている値の形（[osc-protocol.md](osc-protocol.md)）。位置の大きさが 0.30 m を超えていないか |
