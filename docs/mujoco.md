# MuJoCo setup

TouchDesigner の送信用プロジェクト（`td/transform_sender.toe`）から、MuJoCo の OpenArm v2 Pedestal のモデルを動かします。実機の左右の Enable は、MuJoCo では使いません。

## 環境を作る

リポジトリのルートで実行します。仮想環境はルートの `.venv` に作られ、TouchDesigner の Python とは独立しています。依存のバージョンは `uv.lock` で固定しています。

```bash
uv sync --extra sim
```

## 起動する

起動用のスクリプトは、物理計算あり・衝突なし・画面 60 FPS で起動します。設定はその起動の間だけ有効で、シェルには残りません。

| OS | コマンド |
|---|---|
| Windows（PowerShell） | `.\run-mujoco.cmd` |
| Linux・macOS | `bash run-mujoco.sh` |

ログに次の行が出たら、TouchDesigner から操作できます。

```text
[OSC] 127.0.0.1:7000 backend=sim
[MODEL] pedestal; Home preset=forward
```

終了は、起動した端末で `Ctrl+C` を押します。

`--record` を付けると、Follow の間の記録を `config/control.yml` の `record_dir` に書き出します（形式は [hardware.md](hardware.md) の「記録」）。

### 環境変数で細かく決める

スクリプトを使わずに起動するときは、次の環境変数を設定してから `uv run --extra sim dora run dataflow-mujoco.yml` を実行します。

| 環境変数 | 値 | 意味 |
|---|---|---|
| `DORA_OPENARM_MUJOCO_VIEWER` | `60` など／`false` | 画面の FPS。`false` で画面を出さない |
| `DORA_OPENARM_MUJOCO_CTRL` | `true`／`false` | `true`：物理計算で関節の位置サーボを動かす。`false`：IK の答えの関節角度をそのまま反映する（運動学の確認用） |
| `DORA_OPENARM_MUJOCO_ENABLE_COLLISION` | `true`／`false` | 衝突の計算の有無 |

`CTRL=true` で `config/control.yml` の `sim_gravity_compensation: true` のときは、bridge が重力の分の補正を関節の目標に足します（シミュレーションだけの補正）。物理計算では、接触や追従の誤差のために Home が完了しないことがあります。その場合は許容誤差を広げず、ログと姿勢を確認してください。

## TouchDesigner の操作

1. `td/transform_sender.toe` を開き、送り先を **MuJoCo**（127.0.0.1、Port 7000）にする。
2. 入力の切り替え（Input source）を **Slider** にし、位置・回転のスライダーを 0 にする。Enable sending data は OFF にしておく。
3. **Move HOME** を押す。ログに `[right] HOME complete -> READY` と `[left] HOME complete -> READY` が出るまで待つ。
4. **Enable sending data** を ON にする。ログに左右の `FOLLOW start` が出る。
5. 右腕の前後のスライダーを少しだけ動かして戻し、手先が前後に動くことを確かめる。続いて、ほかの軸、回転、グリッパー、左腕を順に確かめる。
6. Enable sending data を OFF にすると、最後の指令の位置で止まる。再開するには、Move HOME から行う。

詳しくは [td-sender.md](td-sender.md)。

## Home の姿勢を選ぶ

`config/control.yml` の `home_preset` で選びます。起動時に一度だけ読むので、変えたら Dora を起動し直します。

| 値 | 姿勢 | 腕の関節角度（J1〜J7、rad） |
|---|---|---|
| `forward`（既定） | 肘を曲げて前へ出す | `[0, 0, 0, 1.570796, 0, 0, 0]` |
| `down` | 腕を下ろす | `[0, 0, 0, 0, 0, 0, 0]` |

- 起動直後の表示の姿勢と、Home の移動先は別です。`down` でも、起動直後は前へ出した姿勢で表示され、Move HOME で腕を下ろします。
- Home の間は、その時点のグリッパーの開き具合を保ちます。
- Follow の位置・回転の基準は、選んだ Home の手先の姿勢です。`down` は腕が伸び切りに近いので、さらに下へ伸ばす目標には届きません。

## Home と Follow の動き

- Home は 4 秒かけて手先の姿勢を補間し、実測の手先が許容範囲（位置 2 cm・姿勢 8°）に入ると完了します。15 秒で届かなければ止めます。
- Home の指令は、0 から 1 に変わったときだけ受け付けます。押し続けても繰り返しません。
- Follow は、Home の完了後に 0 から 1 に変わったときだけ始まります。Home の完了前から ON にしていても、自動では始まりません。
- Follow の速さは、IK の前に手先の位置の速さを `follow_hand_speed_m_s` で、IK の後に各関節の速さを `follow_joint_speed_rad_s` で制限します（`sim` と `hardware` で別の値）。関節は、すべての関節を同じ比率で遅くするので、手先の経路は曲がりません。

## テスト

```bash
uv run --extra sim pytest -q
```

実機を使わずに、bridge の状態の移り変わり、速度の制限、通信の途絶、実機のノードの開始・停止（模擬のドライバー）、OSC → IK → MuJoCo のモデルまでの流れを確かめます。

Dora のプロセスを実際に起動するテストは、環境変数を設定したときだけ動きます。結果は `out/` に書き出します。

```powershell
$env:OPENARM_RUN_DATAFLOW_TESTS = '1'
uv run --extra sim pytest -q tests/test_dataflow.py
uv run --extra sim pytest -q -s tests/test_physics.py
$env:OPENARM_RUN_DATAFLOW_TESTS = '0'
```

Linux・macOS では、`OPENARM_RUN_DATAFLOW_TESTS=1 uv run --extra sim pytest -q tests/test_dataflow.py` のように実行します。
