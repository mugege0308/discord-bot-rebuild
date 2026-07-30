# Discord秘書Bot

Discord上で動作する自分用の秘書Botです。
今後の基本操作は `/` コマンドです。`@初号ちゃん` とメンションすると、ボタン付きの操作パネルを表示します。

## ブルーアーカイブ ダメージ計算（第1段階）

`/ba_damage` は、生徒の攻撃力やスキル倍率、敵防御力などを手動入力し、非会心・会心の想定ダメージ範囲を計算するコマンドです。現段階では生徒DB、Webサイトからの自動取得、TL自動生成は行いません。外部通信や追加の環境変数も使いません。

必須項目:

- `attack`: 基礎攻撃力（0〜1,000,000,000）
- `skill_percent`: **1ヒット当たり**のスキル倍率。800%なら`800`
- `enemy_defense`: 敵の防御力（0〜1,000,000,000）

任意項目:

- `attack_buff_percent`: 攻撃力バフ率。既定`0`
- `damage_buff_percent`: 与ダメージ増加率。既定`0`
- `defense_debuff_percent`: 防御力デバフ率。既定`0`。100%超でも実効防御力は0未満にしない
- `attribute_multiplier`: 属性相性倍率。既定`1.0`
- `terrain_multiplier`: 地形適性倍率。既定`1.0`
- `critical_damage_percent`: 会心ダメージ倍率。既定`200`
- `hit_count`: ヒット数。既定`1`、最大`100`
- `random_min_percent` / `random_max_percent`: ダメージ乱数の最小・最大。既定はいずれも`100`
- `is_critical`: 選択結果を会心として表示するか。既定`false`（非会心・会心の両方は常に表示）

入力例:

```text
/ba_damage attack:10000 skill_percent:800 enemy_defense:1000
```

上記の既定条件での出力例:

```text
実効攻撃力: 10,000
実効防御力: 1,000
防御倍率: 0.625
非会心（1ヒット）: 50,000〜50,000（平均 50,000）
会心（1ヒット）: 100,000〜100,000（平均 100,000）
```

スキル説明に書かれた倍率が複数ヒットの合計値である場合は、ゲーム内仕様を確認し、1ヒット当たりの倍率へ直して入力してください。現在は各ヒットが同じ倍率・補正を受けるものとして合計します。

### 採用している計算モデル

内部の割合は`50% = 0.5`、スキル倍率は`800% = 8.0`へ統一します。

```text
実効攻撃力 = 基礎攻撃力 × (1 + 攻撃力バフ率)
実効防御力 = max(敵防御力 × (1 - 防御力デバフ率), 0)
防御倍率 = 1666.6666666667 ÷ (1666.6666666667 + 実効防御力)
非会心ダメージ = 実効攻撃力 × スキル倍率 × (1 + 与ダメージ増加率)
                 × 防御倍率 × 属性相性倍率 × 地形適性倍率 × ダメージ乱数
会心ダメージ = 非会心ダメージ × 会心ダメージ倍率
```

防御倍率の定数と式は、ブルーアーカイブのプレイヤーコミュニティで広く使われている実測ベースの式を採用しています。地形倍率については攻略情報でも0.8〜1.3倍の補正が案内されています。ただし、ゲーム運営が正式な完全式として公開した資料を確認できていないため、このBotでは**公式確定式ではなく一般的な仮定に基づく近似**として扱います。

確認済み・実装上確定している部分:

- 入力百分率から内部倍率への変換
- バフ・デバフ適用、100%超の防御デバフで実効防御力を0に固定する処理
- NaN、無限大、負数、上限超過、乱数範囲逆転の拒否
- 非会心・会心、乱数最小・最大・平均、複数ヒット合計の算出
- 最終表示での`ROUND_HALF_UP`による整数化

一般的な仮定に基づく部分:

- 防御倍率の`1666.6666666667`定数
- 攻撃力、与ダメージ、属性、地形、会心を上記順序の乗算層として扱うこと
- 全ヒットへ同じ倍率と乱数範囲を適用すること

今後検証が必要な部分:

- ゲーム内部の各段階での切り捨て・四捨五入順序
- 個別スキルの固定値、分割倍率、追加ダメージ、会心可否、特殊バフ
- レベル差、会心抵抗、敵固有の被ダメージ補正やダメージ上限

数式は`services/blue_archive_damage.py`の`calculate_base_damage()`、`calculate_defense_multiplier()`、`calculate_critical_damage()`へ分離してあり、検証結果に合わせて交換できます。参考にした公開情報は[ブルーアーカイブ攻略情報 Wikiのステータス・ダメージ計算](https://wikiwiki.jp/bluearchive/%E8%B1%86%E7%9F%A5%E8%AD%98)および[地形適性倍率の解説](https://game8.jp/blue-archive/639677)です。これらは公式仕様書ではありません。

将来は、この独立エンジンを保ったまま、生徒データベースから攻撃力・スキル倍率を選ぶ入力層と、計算結果を組み合わせるTL作成機能を別レイヤーとして追加する予定です。シャーレDBなど外部サイトの無断転載・スクレイピングは行いません。

## Claude Codeの起動

`PC_CONTROL_ALLOWED_USER_IDS` に登録したユーザーは、Discordから自宅PC上のClaude Code起動用bat/cmdを起動できます。

```text
/claude_start
/claude_status
```

`.env`には起動ファイルの絶対パスを設定します。許可する拡張子は `.bat` と `.cmd` だけです。

```env
CLAUDE_START_FILE=C:\Users\PC_User\Documents\Codex\start_claude_code.bat
```

`start_claude_code.bat`の完成例:

```bat
@echo off
cd /d "C:\Users\PC_User\Documents\Codex"
claude
pause
```

起動処理は `cmd.exe` を引数配列で呼び出し、`shell=False` と `CREATE_NEW_CONSOLE` を使用します。Discordから任意のパスやコマンドを渡すことはできません。起動済みの場合は多重起動しません。

プロセス状態は `data/claude_process_state.json` に原子的に保存します。Bot再起動後はPIDに加えてプロセス名と生成時刻を照合し、停止済みPIDや別プロセスに再利用されたPIDを起動中として扱わないようにします。壊れた状態JSONはログに記録して停止中として扱います。

## Claude Code Remote Control

`PC_CONTROL_ALLOWED_USER_IDS` に登録したユーザーは、外出先のDiscordから、`.env` で固定したプロジェクトのClaude Code Remote Controlを起動・確認・停止できます。

### 事前準備

1. 対象PCへRemote Control対応版のClaude Codeをインストールし、`claude` がPATHから実行できることを確認します。
2. `CLAUDE_REMOTE_PROJECT_DIR` に設定する対象フォルダで一度Claude Codeを直接起動します。
3. Claude Codeへのログインと、そのフォルダのワークスペース信頼の承認を完了します。
4. `.env` を設定してBotを再起動します。

```env
PC_CONTROL_ALLOWED_USER_IDS=123456789012345678
CLAUDE_REMOTE_ENABLED=true
CLAUDE_REMOTE_COMMAND=claude
CLAUDE_REMOTE_PROJECT_DIR=C:\Users\PC_User\Documents\Codex\対象プロジェクト
CLAUDE_REMOTE_SESSION_NAME=初号ちゃん
CLAUDE_REMOTE_START_TIMEOUT_SECONDS=30
CLAUDE_REMOTE_STOP_TIMEOUT_SECONDS=10
```

`CLAUDE_REMOTE_COMMAND` にはPATH上の `claude`、または管理者が決めた実行ファイルの固定パスを指定します。Windowsのnpmインストールで `claude.cmd` が解決された場合にも対応します。Discordからパス、コマンド、追加引数、セッション名を指定することはできません。

### Discordコマンド

```text
/claude_remote_start   Remote Controlを起動し、取得できた接続URLをephemeralで表示
/claude_remote_status  PID、起動時刻、経過時間、対象フォルダ、URL取得状況、終了コード、エラー概要を表示
/claude_remote_link    取得済み接続URLをephemeralで再表示
/claude_remote_stop    通常終了、terminate、killの順で安全に停止
```

接続できるのはRemote Controlプロセスが動いている間だけです。接続URLは本人だけに見えるephemeral応答で扱い、公開チャンネルへ貼らないでください。Botは接続URLやClaude Codeの全出力を通常ログへ記録しません。

状態はMVPとしてBotのメモリ内だけに保持します。Botを再起動すると接続URLを失うため、その場合はClaudeアプリまたは `https://claude.ai/code` のCode一覧から `CLAUDE_REMOTE_SESSION_NAME` のセッションを探してください。Bot再起動後に古いPIDだけを使ってプロセスを停止することはありません。

### Remote Controlのトラブルシューティング

- 「機能は無効」: `CLAUDE_REMOTE_ENABLED=true` にしてBotを再起動します。
- 「対象プロジェクトが存在しない」: `CLAUDE_REMOTE_PROJECT_DIR` の絶対パスとフォルダの存在を確認します。
- 「Claude Codeコマンドが見つからない」: 対象PCで `claude --version` を実行し、PATHまたは `CLAUDE_REMOTE_COMMAND` を確認します。
- 「未ログイン」: 対象フォルダでClaude Codeを直接起動してログインします。
- 「ワークスペース信頼が未承認」: 対象フォルダでClaude Codeを直接起動し、信頼を承認します。
- 「Remote Controlを利用できない」: Claude Codeを更新し、利用中の環境・契約でRemote Controlが提供されているか確認します。
- 「起動タイムアウト」: プロセスは起動を継続している場合があります。少し待って `/claude_remote_status` と `/claude_remote_link` を実行します。
- URL取得前に終了する: 対象PCで同じフォルダから `claude remote-control --name "初号ちゃん"` を直接実行し、表示されるログイン・信頼・機能提供エラーを確認します。
- 停止に失敗する: 対象PCのタスクマネージャーで、Botが起動したClaude Codeプロセスの状態を確認します。

## 開発タスク機能（Phase 4A-v2・隔離コピーによるClaude Code試験実行）

`DEV_TASK_ALLOWED_USER_IDS` に登録したユーザーは、Discordから開発タスクを起票できます。`PC_CONTROL_ALLOWED_USER_IDS` とは別の許可リストです。

```text
/dev_task task:内容 engine:auto base_branch:任意
/dev_task_status task_id:任意（省略時は自分の最新タスク）
/dev_task_stop
```

`/dev_task` で起票すると、内容に応じてCodex/Claudeへの割り振りが判定され、承認ボタン付きのメッセージが表示されます。**現時点では、承認してもCodex・Claudeのどちらのエンジンも実際には実行されません。** 「承認して実行」を押すと、エンジンによらず安全な理由付きで失敗として記録されます（worktreeもサンドボックスも一切作成されません）。これは意図的な暫定状態です:

- 旧方式（タスク専用git worktree内でClaude Codeを実行する方式）は廃止し、二度と実Claude実行には使いません。
- 新方式（後述する隔離コピー・サンドボックス方式）は実装済みですが、この`/dev_task`承認フローへはまだ接続していません。理由は、結果ビューの「保持」「破棄」ボタンが今もgit worktreeの削除（`services.dev_worktree.remove_worktree`）だけを呼び出す設計のためです。中途半端に接続すると、サンドボックスに対して「破棄」を押してもサンドボックス自体は削除されず、Storeにだけ「破棄済み」と記録される、気づきにくい放置バグを生みます。この接続（結果ビューをサンドボックス対応させること）は今後のフェーズの課題です。

### 隔離コピー（isolated copy）によるClaude Code試験実行

`/dev_task`とは別に、Claude Codeをheadlessで安全に試験実行するための独立したモジュール一式があります（現時点ではDiscordから直接は呼び出せず、手動/スクリプトから利用する想定です）。

1. `services.dev_copy_workspace.create_sandbox(task_id)` が、本体リポジトリから許可された拡張子・ファイル名だけを`DEV_TASK_SANDBOX_ROOT/<task_id>`へコピーします。`.git`・`.env`（`.env.example`を除く）・`data`/`outputs`/`logs`/`work`・認証情報・キャッシュ・`.claude`/`.codex`など本体固有のディレクトリは除外します。symlink/junction/reparse pointは追跡もコピーもしません。ハードリンク共有がないこと、本体と独立した実体であることをコピー時・検証時それぞれで確認します。`DEV_TASK_SANDBOX_ROOT`が本体リポジトリと同一・その配下・その親（結果的に本体を包含する配置）になっている場合は、`Path.resolve()`で正規化したうえで（大文字小文字や区切り文字の違い、表記ゆれも吸収します）コピー開始前に安全に拒否します（`create_sandbox`だけでなく、Claude起動直前に再検証する`services.claude_headless.run`側の`verify_sandbox_safe`、および`discard_sandbox`でも同じ判定が働きます）。
2. `services.claude_headless.run(task_id=..., prompt_text=..., timeout_seconds=...)` が、Claude起動の直前にサンドボックスを再検証し、そのサンドボックスだけをカレントディレクトリとしてClaude Code CLIを1回だけheadless実行します。タスク内容は標準入力で渡し、シェル展開やコマンド連結は一切行いません。
3. 権限は`--permission-mode dontAsk`＋`--tools Read,Edit,Write,Glob,Grep`（Bash/PowerShell/WebFetch/WebSearch/MCP/slash command/browser連携はすべて利用不可）に固定しています。`--dangerously-skip-permissions`/`bypassPermissions`は使用しません。`--bare --strict-mcp-config --disable-slash-commands --no-chrome --no-session-persistence`も付与します。
4. Botプロセスの環境変数は継承しません。`SystemRoot`/`WINDIR`/`PATH`/`TEMP`/`TMP`のみの許可リストから環境変数を構築します。`DISCORD_TOKEN`やAPIキー・トークン・AWS/GitHub/SSH認証情報などは子プロセスへ絶対に渡しません。Claude自体の認証にAPIキーが必要な場合は`CLAUDE_HEADLESS_ANTHROPIC_API_KEY`を専用に設定してください（Botの他の環境変数からの継承はしません）。
5. 変更確認はgitを使わず、コピー時に記録したSHA256ハッシュとの比較で行います（`services.dev_copy_workspace.diff_sandbox_against_baseline`）。
6. Claude実行中またはプロセス終了未確認の間は、サンドボックスの破棄を拒否します（`services.dev_copy_workspace.discard_sandbox`）。破棄可能になるのは、OSプロセス終了・子プロセス終了確認・標準入力書き込みスレッド終了・標準出力/標準エラー読み取りスレッド終了・結果保存がすべて完了した後だけです。
7. 本体への自動反映・自動commit・自動merge・自動patch適用は一切行いません。人間が変更内容を確認したうえで、保持するか破棄するかを判断してください。

`.env`設定:

```env
CLAUDE_HEADLESS_ENABLED=false
CLAUDE_CLI_PATH=C:\Users\PC_User\.local\bin\claude.exe
CLAUDE_HEADLESS_TIMEOUT_SECONDS=1800
CLAUDE_HEADLESS_MAX_OUTPUT_BYTES=1048576
CLAUDE_HEADLESS_ANTHROPIC_API_KEY=
CLAUDE_HEADLESS_WORKSPACE_MODE=isolated_copy
DEV_TASK_SANDBOX_ROOT=C:\Users\PC_User\Documents\Codex\dev-task-sandboxes
DEV_TASK_SANDBOX_MAX_FILES=5000
DEV_TASK_SANDBOX_MAX_TOTAL_BYTES=524288000
```

`CLAUDE_HEADLESS_ENABLED`が未設定・不正な値の場合は無効として扱われ、`CLAUDE_HEADLESS_WORKSPACE_MODE`が`isolated_copy`以外の場合や`DEV_TASK_SANDBOX_ROOT`が未設定の場合も安全に拒否されます。いずれもBot全体の起動には影響しません。`CLAUDE_CLI_PATH`が未設定・存在しない・ディレクトリ・許可されていない拡張子（Windowsでは`.exe`/`.cmd`/`.bat`のみ）の場合も同様です。`--bare`使用によりOAuth/keychain認証は使えないため、実際にClaude Codeを試験実行する前に、ログイン済みのOAuthセッションではなく`CLAUDE_HEADLESS_ANTHROPIC_API_KEY`（APIキー）を用意する必要があります。

結果はDiscordの15分の応答制限に依存せず、起票者へのDM、それが失敗した場合は`DEV_TASK_RESULT_CHANNEL_ID`で指定したチャンネルへ通知します。どちらも失敗した場合はログにのみ記録され、Bot自体は落ちません。結果メッセージには「保持」「破棄」ボタンがあり、破棄を選ぶとworktreeを削除します（現時点では上記のとおりworktree専用で、サンドボックスには未接続です）。別ブランチのworktreeが登録されている場合や安全確認に失敗した場合は、破棄ボタンを押しても強制削除しません。

承認ボタン・結果ボタンはどちらも起票者本人だけが操作できます。**Bot再起動後、これらのボタンは無効になります。** 再起動時に実行中だったタスクは自動的に失敗として記録されますが（Phase 1の起動時リカバリ）、承認待ちのまま再起動をまたいだタスクは、そのボタンが使えなくなるため`/dev_task_status`で状態を確認したうえで、必要なら`/dev_task`から新たに起票し直してください。

## VOICEVOXによるVC読み上げ

ローカルで起動したVOICEVOXを使い、Discordのボイスチャンネルで指定文を読み上げられます。

```text
/vc_join
/vc_say text:こんにちは、初号ちゃんです
/vc_leave
```

- `/vc_join` は実行者が参加中のVCへ接続します。別VCに接続中なら移動します。
- `/vc_say` は最大200文字をVOICEVOXでwavへ変換し、guildごとのキューへ追加します。Botが未接続なら実行者のVCへ自動接続します。
- `/vc_leave` は再生を停止し、待機キュー、ワーカー、一時wavを破棄して退出します。
- Bot終了時や手動切断時にも、キューと一時ファイルを可能な範囲でクリーンアップします。
- 今回は自動読み上げ、Webhook/Bot発言の読み上げ、音声受信、録音、音声認識を実装していません。

`.env`設定:

```env
VOICEVOX_API_URL=http://127.0.0.1:50021
VOICEVOX_SPEAKER_ID=1
VOICEVOX_REQUEST_TIMEOUT_SECONDS=30
VC_SAY_MAX_LENGTH=200
```

`VOICEVOX_API_URL` は `localhost`、`127.0.0.1`、`::1` のHTTP URLだけ利用できます。音声再生にはFFmpegと、`discord.py[voice]` が導入するPyNaCl・daveyが必要です。依存関係は次で導入できます。

```bash
python -m pip install -r requirements.txt
```

VOICEVOXを起動し、HTTP APIが `http://127.0.0.1:50021` で待機している状態で使用してください。VOICEVOXが停止中、HTTPエラー、タイムアウト、JSON異常の場合はBotを落とさずDiscordへエラーを返します。一時音声は `data/voice_temp` にランダム名で作成され、再生成功・失敗のどちらでも削除されます。

## 選択式キャラクター生成

`/character_generate` では、髪色・髪型・目の色・表情・服装・背景・雰囲気をSelectメニューから選んでキャラクター画像を生成できます。各項目には「指定なし」と「ランダム」があり、2ページの選択後に、選択内容と組み立てた日本語を最終確認できます。

確認画面の「生成する」は、既存の `/generate` と同じLM Studio、Memory、ComfyUI、画像選別AI処理を使います。「作り直す」で最初の選択画面へ戻れます。

```text
/character_generate
/character_generate reference_id:1
```

`reference_id` を指定した場合は、指定した画像参考Memoryの全タグをプロンプト生成時に利用します。選択肢は `data/character_template_options.json` で追加・変更できます。DiscordのSelectメニュー上限に合わせ、1カテゴリ25項目以内にしてください。

## 画像参考Memoryの名前

画像解析時に、`note` とは別に任意の名前を付けられます。名前は1〜50文字で、前後空白・改行・制御文字は保存時に除去されます。`name` を省略した場合は、「Memoryに保存」を押すとVision解析タグから作った自動名入りのModalが開き、そのまま保存するか編集できます。

```text
/image_analyze image:画像 name:黒髪ボブ緑目
/image_memory_rename id:3 name:銀髪小悪魔
```

`/image_memory_list` は「ID / 名前 / 短い要約」を表示し、`/image_memory_show` は名前を先頭に表示します。`/character_generate reference_id:3` の選択画面と確認画面にも、IDと名前を表示します。

保存先の `data/image_memory/ref_XXXX.json` には `name` を保存します。互換性のため `title` も同じ表示名で更新し、古いデータは `name`、`title`、`参考画像 #ID` の順で表示名を解決します。ファイル名とIDは改名しても変わりません。

### 評価と生成利用

画像参考Memoryごとに、好み度と通常生成の自動候補に使うかを管理できます。

```text
/image_memory_rate id:3 rating:5
/image_memory_use id:3 enabled:false
```

- `rating`: 0〜5。新規保存時は0（未評価）で、コマンドでは1〜5を指定します。
- `use_for_generation`: `true` / `false`。新規保存時は`true`です。
- 古いJSONにフィールドがない場合は、`rating=0`、`use_for_generation=true`として扱います。
- 通常の `/prompt`、`/generate`、`/character_generate` は、生成利用がONの関連Memoryだけを候補にし、評価の高いものを優先します。
- 通常参照では画像Memory由来タグを最大5個に制限します。
- `reference_id` を明示した場合は、評価や生成利用ON/OFFに関係なく、所有者本人の参考画像を使用できます。

保存JSONでは、将来画像選別AIへ評価データを渡せるよう独立フィールドとして保持します。現時点では画像選別AIの再学習には使用しません。

```json
{
  "rating": 0,
  "use_for_generation": true
}
```

### 元画像の保存と表示

`/image_analyze` の「Memoryに保存」を押すと、解析結果と一緒に解析対象の元画像を保存します。8MB制限とJPEG / PNG / WebP制限は従来どおりです。

```text
data/image_memory/ref_0003.json
data/image_memory/images/ref_0003.webp
```

JSONには対応する画像パスを保存します。

```json
{
  "image_path": "data/image_memory/images/ref_0003.webp"
}
```

- `/image_memory_list` は1ページ3件で、ID、名前、評価、生成利用ON/OFF、短い要約、元画像サムネイルを表示します。
- 別ページは `/image_memory_list page:2` のように指定します。
- `/image_memory_show id:3` は解析内容とSDタグに加えて元画像を添付します。
- `image_path` がない古いデータは「元画像なし」と表示します。
- `/image_memory_delete` は専用JSON、保存画像、linked通常Memoryをまとめて削除します。
- 保存画像は `data/image_memory/images` 内のIDと一致する固定ファイル名だけを読み書き・削除対象にします。

## セットアップ

### 1. Pythonを確認する

```bash
python --version
```

`python` が見つからない場合:

```bash
py --version
```

### 2. 依存関係をインストールする

```bash
pip install -r requirements.txt
```

または

```bash
py -m pip install -r requirements.txt
```

Windowsで `ZoneInfoNotFoundError` や `No module named 'tzdata'` が出た場合:

```bash
python -m pip install -r requirements.txt
```

### 3. `.env` を作成する

PowerShell:

```powershell
Copy-Item .env.example .env
```

コマンドプロンプト:

```cmd
copy .env.example .env
```

### 4. `.env` を設定する

```env
DISCORD_BOT_TOKEN=ここにBotトークンを貼り付ける
DISCORD_GUILD_ID=テスト用DiscordサーバーID

WEATHER_CHANNEL_ID=起動時の天気通知を送るチャンネルID
WEATHER_LATITUDE=35.0116
WEATHER_LONGITUDE=135.7681
WEATHER_LOCATION_NAME=京都
WEATHER_NOTIFY_AFTER_HOUR=6
WEATHER_EVENING_NOTIFY_AFTER_HOUR=18

PC_CONTROL_ALLOWED_USER_IDS=123456789012345678
PC_SHUTDOWN_DELAY_SECONDS=60
```

## 起動方法

### Windowsでダブルクリック起動

botフォルダ直下の `start_bot.bat` をダブルクリックすると起動できます。

`start_bot.bat` は以下を行います。

- botフォルダをカレントディレクトリにする
- `chcp 65001` でUTF-8表示に切り替える
- `python main.py` を実行する
- `python` が失敗した場合は `py main.py` を試す
- 最後に `pause` して、エラー内容を確認できるようにする

### Windowsログイン時に自動起動する

`start_bot.bat` のショートカットをWindowsのスタートアップフォルダに入れると、次回Windowsログイン時にBotを自動起動できます。

手順:

1. botフォルダ内の `start_bot.bat` を右クリック
2. `ショートカットを作成` を選ぶ
3. `Win + R` を押す
4. `shell:startup` と入力してEnter
5. 開いたスタートアップフォルダに、作成したショートカットを移動
6. 次回Windowsログイン時にBotが自動起動する

注意:

- Windowsにログインした後に起動します
- 開いた黒い画面を閉じるとBotも止まります
- `.env` や `commands` フォルダがあるbotフォルダ内の `start_bot.bat` を使ってください
- `start_bot.bat` 本体ではなく、ショートカットをスタートアップフォルダに置くのがおすすめです

### PowerShellで起動

通常:

```bash
python main.py
```

`python` が使えない場合:

```bash
py main.py
```

### 開発用の自動再起動

Pythonファイルを保存したときにBotを自動再起動したい場合は、開発用ランナーを使います。

```bash
python dev_runner.py
```

`python` が使えない場合:

```bash
py dev_runner.py
```

`dev_runner.py` は開発用です。本番環境や常時稼働では、通常通り `python main.py` を使ってください。
Botトークンや `.env` の読み込みは、これまで通り `main.py` 側で行います。

## 基本の使い方

このBotは `/` コマンドで操作します。

主なコマンド:

- `/help`
- `/ping`
- `/memo`
- `/memo_list`
- `/memo_delete`
- `/todo`
- `/todo_add`
- `/todo_list`
- `/todo_done`
- `/todo_delete`
- `/today`
- `/progress`
- `/log_list`
- `/log_delete`
- `/daily_report_set`
- `/daily_report_off`
- `/daily_report_now`
- `/remember`
- `/memory_list`
- `/memory_delete`
- `/chat`
- `/weather`
- `/nisa`
- `/pc_shutdown`
- `/pc_shutdown_status`
- `/pc_shutdown_cancel`
- `/bot_restart`
- `/jockie_shuffle`
- `/playlist_save`
- `/playlist_list`
- `/playlist_delete`
- `/jockie_shuffle_saved`
- `/double_translate_start`
- `/double_translate_status`
- `/double_translate_stop`
- `/health_check`
- `/prompt`
- `/generate`
- `/generate_raw`
- `/generate_ai`
- `/image_analyze`
- `/image_memory_list`
- `/image_memory_show`
- `/image_memory_delete`

以前の `@初号ちゃん help` のようなメンションコマンドは旧方式です。今後は `/` コマンドを使ってください。

## 操作パネル

`@初号ちゃん` とメンションすると、カテゴリ型の操作パネルが表示されます。

```text
初号ちゃんです。
何をしますか？

[今日] [記録] [便利]
[音楽] [PC操作] [ヘルプ]
```

メインカテゴリ:

- 今日
- 記録
- 便利
- 音楽
- PC操作
- ヘルプ

各カテゴリを押すと、詳細メニューが本人だけに表示されます。

カテゴリごとの内容:

- 今日: 今日のまとめ、今日の天気、明日の天気、傘確認
- 記録: メモ一覧、メモの使い方、ToDo一覧、ToDoの使い方、メモ/TODO/進捗追加、最近の記録、日次レポート操作
- 便利: 二重翻訳の開始/状態確認/停止、NISA目安
- 音楽: 保存済み再生リスト一覧、Jockie Music補助の使い方
- PC操作: シャットダウン予約、シャットダウン取消、使い方
- ヘルプ: `/help` 相当の説明

注意:

- パネルは呼び出した本人用です
- 他の人が押しても操作できません
- パネルが古くなって反応しない場合は、もう一度 `@初号ちゃん` と送ってください
- メインパネルはカテゴリ入口です。カテゴリを押しても元のメインパネルは消えません
- メモ追加、TODO追加、進捗追加は記録パネルのModalからも入力できます
- TODO完了・削除、プレイリスト保存のようにIDや名前の指定が必要な操作は `/` コマンドを使ってください
- PC操作は危険操作のため、許可ユーザーだけが実行でき、実行前に確認ボタンが表示されます
- メンション後に文章がある場合も、基本的には操作パネルを表示します

## スラッシュコマンド

### /help

使える機能を表示します。

### /ping

Botが起きているか確認します。

### /memo

```text
/memo content:給料日は25日
```

メモを保存します。

### /memo_list

SQLiteの `records` に保存された最近のメモを表示します。

### /memo_delete

```text
/memo_delete number:3
```

指定した記録IDのメモを削除します。

### ToDo機能

やることをユーザーごとに保存できます。

使い方:

```text
/todo content:定期更新する
/todo_add content:定期更新する
/todo_list
/todo_done todo_id:3
/todo_delete number:1
```

`/todo` と `/todo_add` は、どちらもSQLiteの `records` に `type=todo` として保存します。
`/todo_add` は既存互換のため、従来の `data/todos.json` にも保存します。
`/todo_list` では未完了TODOだけを表示し、表示には `#3` のような `records.id` が含まれます。
`/todo_done` は、そのIDを指定して完了にします。

### 今日のまとめ

`/today` で、今日の天気、未完了ToDo、メモ状況をまとめて確認できます。

```text
/today
```

NISA目安は自動では含めません。
必要な場合は `/nisa` を使ってください。

### 記録・進捗・日次レポート

メモ、TODO、進捗をSQLiteの `data/daily_log.db` に保存します。

使い方:

```text
/memo content:気づいたこと
/todo content:Botの権限設定を確認する
/progress content:日次レポート機能を追加した
/todo_list
/todo_done todo_id:3
/log_list kind:all limit:10
/log_delete record_id:3
```

`/memo` は既存の `data/memos.json` にも保存しつつ、日次レポート用にSQLiteの `records` にも保存します。
新しい記録確認は `/log_list`、削除は `/log_delete` を使います。

日次レポート:

```text
/daily_report_set channel:#作業ログ hour:7 minute:0
/daily_report_off
/daily_report_now date:2026-06-28
```

- `daily_report_set` は、サーバーごとに投稿先チャンネルと投稿時刻を保存します
- 自動投稿は1分ごとに現在時刻を確認し、設定時刻に昨日分を投稿します
- `last_posted_for` に対象日を保存し、同じ日付の二重投稿を防ぎます
- `daily_report_now` の `date` を省略すると、昨日分を今すぐ投稿します
- FeatherPanelなどでBotを再起動しても、`data/daily_log.db` が残っていれば記録と設定は残ります

操作パネルからも使えます。

`@初号ちゃん` → `記録` から以下を操作できます。

- メモ追加
- TODO追加
- 進捗追加
- 未完了TODO一覧
- 最近の記録一覧
- 日次レポート手動表示
- 現在のチャンネルを7:00の投稿先に設定
- 投稿時刻を入力して設定
- 日次レポート停止

メモ追加、TODO追加、進捗追加はModalで内容を入力します。

### 簡易メモリ機能

LM Studioのモデル自体を学習させるのではなく、ユーザーごとの好み・設定・会話メモを `data/memory.json` に保存します。

使い方:

```text
/remember text:画像は淡い色、眠たげな雰囲気が好き
/memory_list
/memory_delete id:1
/chat text:この設定を踏まえて短く相談に乗って
```

動作:

- メモリはDiscordユーザーごとに分けて保存します。
- `/chat` と `/prompt` 実行時に、関連しそうな保存済みメモリを「このユーザーの好み」として自然な文章にまとめ、system promptへ追記します。
- 「好き」「よく使う」「お気に入り」「好み」などを含むメモリは優先度を高く扱います。
- 「画像」「プロンプト」「絵」などの依頼では、画像生成関連のメモリを優先します。
- `/chat` では、可能な範囲で保存済みメモリを引用・活用するようLM Studioへ伝えます。
- `/prompt` では、保存済みメモリを初号ちゃん側でルールベースのタグに変換し、画風・色・構図・タグ選びへ反映するようLM Studioへ伝えます。
- `/prompt` と `/generate` では、メモリを「常時反映してよい好み」と「入力文に関連語がある時だけ反映する要素」に分けて扱います。
- 検索は全文・単語一致の簡易検索です。RAGやベクトル検索は使っていません。
- 既存の `/memo` とは別機能です。

メモリからタグへの変換例:

```text
暖色が好き -> warm lighting, orange tones
柔らかい光が好き -> gentle lighting
自然な目が好き -> detailed eyes, expressive eyes
```

常時反映してよい好み:

```text
可愛い系、人物品質、自然な目、暖色傾向、柔らかい光、淡い色
```

入力文に関連語がある時だけ反映する要素:

```text
神社、鳥居、和服、制服、カフェ、雨、サンタ服、桜、夕焼け、メイド
```

たとえば、メモリに「神社が好き」があっても、入力文に「神社」や「鳥居」がない場合は `shrine` / `torii` を足しません。
メモリに「和服が好き」があっても、入力文に「和服」や「着物」がない場合は `kimono` / `obi` を足しません。
背景指定がない入力では、背景系メモリを無理に追加しないようにしています。
採用したMemoryタグと除外したMemoryタグはログに出力します。

### /weather

```text
/weather day:今日
/weather day:明日
```

今日または明日の天気を表示します。

### /nisa

NISAのざっくり目安を表示します。
投資判断そのものではなく、年間枠や毎月金額の目安を確認するための表示です。

### /jockie_shuffle

YouTube再生リストURLから、Jockie Musicでシャッフル再生しやすいコピー用コマンドを作ります。
BotがJockie Musicを直接操作するのではなく、ユーザーがコピーして送信するための文字列を返します。

```text
/jockie_shuffle url:https://www.youtube.com/playlist?list=xxxxx
```

返信例:

```text
Jockie Musicでシャッフル再生するなら、以下をコピーして送信してください。
```

コピーするコマンド:

```text
m!play https://www.youtube.com/playlist?list=xxxxx --shuffle
```

Jockie Musicのプレフィックスが `m!` 以外の場合は、`prefix` を指定してください。

```text
/jockie_shuffle url:https://www.youtube.com/playlist?list=xxxxx prefix:!
```

### 再生リスト保存機能

よく使うYouTube再生リストを名前付きで保存できます。

例:

```text
/playlist_save name:作業用 url:https://www.youtube.com/playlist?list=xxxxx
/playlist_list
/jockie_shuffle_saved name:作業用
```

出力されたJockie Music用コマンドをコピーして、自分で送信してください。

削除する場合:

```text
/playlist_delete name:作業用
```

注意:

- 初号ちゃんがJockie Musicを直接操作する機能ではありません
- Jockie MusicがBotの発言に反応しない可能性があるため、ユーザーがコピーして貼る方式です
- 保存データは `data/playlists.json` に保存されます
- 保存データはユーザーごとに分かれます

## 二重翻訳機能

指定した時間だけ、自分の発言を `日本語→英語→日本語` に変換してBotが返信します。

使い方:

```text
/double_translate_start minutes:5
/double_translate_status
/double_translate_stop
```

動作:

- 対象は開始したユーザー本人のみです
- 他ユーザーのメッセージは翻訳されません
- Bot再起動時に状態はリセットされます
- 500文字を超える文章は翻訳せず、短くするよう案内します
- 翻訳に失敗する場合があります

例:

```text
今日は仕事がだるい
```

```text
今日の仕事は面倒です。
```

## PCシャットダウン機能

許可したDiscordユーザーだけが、DiscordからBotが起動しているPCのシャットダウンを予約できます。
危険な操作なので、許可ユーザーIDを `.env` に設定した場合だけ使ってください。

使い方:

```text
/pc_shutdown
/pc_shutdown_status
/pc_shutdown_cancel
```

設定:

```env
PC_CONTROL_ALLOWED_USER_IDS=123456789012345678
PC_SHUTDOWN_DELAY_SECONDS=60
```

動作:

- `/pc_shutdown` を実行すると、確認ボタンが表示されます
- 「実行する」を押すまで実行されません
- 確認ボタンを押せるのは、`/pc_shutdown` を実行した本人だけです
- Windowsでは `shutdown /s /t 秒数` を固定コマンドとして実行します
- `/pc_shutdown_status` で、Botが最後に保存した予約内容を確認できます
- `/pc_shutdown_cancel` で予約済みシャットダウンをキャンセルできます

注意:

- Botが起動しているPCだけをシャットダウンできます
- PCがスリープ中・電源OFF中は操作できません
- 許可ユーザーIDを `.env` の `PC_CONTROL_ALLOWED_USER_IDS` に設定してください
- 複数ユーザーを許可する場合は、カンマ区切りで指定します
- Windows以外では実行しません
- 任意のOSコマンドを実行する機能ではありません
- `data/pc_shutdown_state.json` には、Botが最後に予約した内容だけを保存します
- Windows側に実際の予約が残っているかを厳密に検出するものではありません
- PC自体を再起動した場合、Windows側のshutdown予約は消える可能性があります

### 操作パネルからPCをシャットダウンする

`@初号ちゃん` → `PC操作` から、シャットダウン予約や取り消しができます。

PC操作パネル:

```text
PC操作メニューです。
許可されたユーザーだけが実行できます。

[シャットダウン] [時間指定シャットダウン] [予約状況]
[シャットダウン取消] [使い方]
```

注意:

- 許可ユーザーだけが操作できます
- 実行前に確認ボタンが出ます
- 予約状況では、Botが最後に保存した予約内容を確認できます
- Botが起動しているPCだけをシャットダウンできます
- PCがスリープ中や電源OFF中は操作できません
- 誤操作防止のため、メインパネルから直接シャットダウンはされません

## 起動時の自動通知

Botを起動した時間帯によって、自動通知の内容が変わります。

- 6:00〜17:59 に起動: 今日の天気を通知
- 18:00以降に起動: お疲れさまメッセージと明日の天気を通知

朝通知と夕方通知は別々に1日1回だけ送信されます。
起動時の自動通知では、今日または明日の天気を表示します。
NISA目安を確認したい場合は `/nisa` を使ってください。

送信条件:

- 現在時刻が日本時間の通知開始時刻以降
- その日の同じ種類の通知をまだ送っていない
- `.env` に `WEATHER_CHANNEL_ID` が設定されている
- 天気APIの取得に成功している

通知済みの日付は `data/weather_state.json` に保存されます。

保存例:

```json
{
  "morning_last_notified_date": "2026-06-24",
  "evening_last_notified_date": "2026-06-24"
}
```

よくある例:

- 朝にPCをつけた場合: 今日の天気が送られます
- 仕事後の18時以降にPCをつけた場合: お疲れさまメッセージと明日の天気が送られます
- 朝にも起動して、夜にも起動した場合: 朝は今日の天気、夜は明日の天気がそれぞれ1回ずつ送られます

## Discord Developer Portalで必要な設定

1. [Discord Developer Portal](https://discord.com/developers/applications) を開きます。
2. `New Application` からアプリケーションを作成します。
3. 左メニューの `Bot` を開き、`Add Bot` でBotを作成します。
4. `Token` からBotトークンを発行し、`.env` の `DISCORD_BOT_TOKEN` に設定します。
5. `Privileged Gateway Intents` で `MESSAGE CONTENT INTENT` を有効にします。
6. 左メニューの `OAuth2` → `URL Generator` を開きます。
7. `Scopes` で `bot` と `applications.commands` を選択します。
8. `Bot Permissions` で `View Channels`、`Send Messages`、`Read Message History` を選択します。
9. 生成されたURLを開き、Botを自分のDiscordサーバーに招待します。

すでにBotを招待済みで `/` にコマンドが出ない場合は、`applications.commands` を含めた招待URLで再招待してください。

## チャンネルIDの取り方

1. Discordのユーザー設定を開く
2. 詳細設定を開く
3. 開発者モードをONにする
4. 天気通知を送りたいチャンネルを右クリック
5. IDをコピー
6. `.env` の `WEATHER_CHANNEL_ID` に貼り付ける

## よくあるトラブル

### start_bot.bat で起動できない

- `python --version` または `py --version` が動くか確認する
- `python -m pip install -r requirements.txt` を実行する
- `.env` が作成されているか確認する
- `DISCORD_BOT_TOKEN` が設定されているか確認する

### 起動しても天気が送られない

- 6時前ではないか
- 今日すでに送信済みではないか
- `WEATHER_CHANNEL_ID` が設定されているか
- Botにそのチャンネルへの送信権限があるか

### Windowsで Asia/Tokyo が見つからない

- `requirements.txt` に `tzdata` があるか確認する
- `python -m pip install -r requirements.txt` を実行する

### `/` コマンドが古いと言われる

- 数分待つ
- Discordを再読み込みする
- `DISCORD_GUILD_ID` が正しいか確認する
- 起動時のguild syncログを確認する

## ファイル構成

```text
secretary-bot/
├─ main.py
├─ daily_log.py
├─ start_bot.bat
├─ commands/
│  ├─ __init__.py
│  ├─ chat.py
│  ├─ help.py
│  ├─ memo.py
│  ├─ ping.py
│  ├─ today.py
│  └─ todo.py
├─ services/
│  ├─ __init__.py
│  ├─ nisa.py
│  ├─ pc_control.py
│  ├─ double_translate.py
│  ├─ jockie.py
│  ├─ translator.py
│  └─ weather.py
├─ views/
│  ├─ __init__.py
│  ├─ panel.py
│  └─ pc_control.py
├─ data/
│  ├─ daily_log.db
│  ├─ image_memory/
│  ├─ memory.json
│  ├─ memos.json
│  ├─ pc_shutdown_state.json
│  ├─ playlists.json
│  ├─ todos.json
│  └─ weather_state.json
├─ .env.example
├─ requirements.txt
└─ README.md
```

`data/daily_log.db` は、記録・TODO・進捗・日次レポート設定を保存するSQLiteデータベースです。
`data/image_memory/` は、画像参考メモの専用JSONを保存します。
`data/memory.json` は、ユーザーIDごとの簡易メモリを保存します。
`data/memos.json` は、ユーザーIDごとのメモを保存します。
`data/pc_shutdown_state.json` は、Botが最後に保存したPCシャットダウン予約内容を保存します。
`data/playlists.json` は、ユーザーIDごとの保存済みYouTube再生リストを保存します。
`data/todos.json` は、ユーザーIDごとのToDoを保存します。
`data/weather_state.json` は、起動時天気通知の送信済み日付を保存します。

これらのデータファイルは、必要になったときに自動作成されます。

## ローカル画像生成操作

Discordから家PC上のStability Matrix / ComfyUI APIを操作するための機能です。通常操作はスラッシュコマンドで行います。

使えるコマンド:

- `/sd_start`
  - `SD_BACKEND=comfy` の場合は `.env` の `COMFYUI_START_FILE` に設定した bat/cmd/exe を起動します。
  - それ以外の場合は `.env` の `SD_START_FILE` に設定した bat/cmd/exe を起動します。
- `/sd_status`
  - `SD_API_URL` のAPIが応答しているか確認します。ComfyUIの場合は通常 `http://127.0.0.1:8188` を使います。
- `/sd_generate prompt:内容`
  - ComfyUI API にWorkflowを投げ、生成画像をDiscordに返します。
- `/sd_latest`
  - 最新生成画像をDiscordに送ります。
- `/sd_save_favorite`
  - 最新画像を `data/sd_favorites` にコピーします。
- `/sd_shutdown_after_done`
  - 次の画像生成完了後にPCシャットダウンを予約します。

必要な `.env` 設定:

```env
SD_BACKEND=comfy
SD_API_URL=http://127.0.0.1:8188
COMFYUI_START_FILE=C:\Users\PC_User\Documents\Codex\2026-06-23\discord-bot-python-discord-py-pc\start_comfyui.bat
COMFY_WORKFLOW_FILE=workflows/comfy_workflow.json
SD_MAX_COUNT=10
SD_MAX_WIDTH=1024
SD_MAX_HEIGHT=1024
SD_MAX_STEPS=60
SD_MAX_CFG=20
```

安全上の注意:

- 実行できるのは `PC_CONTROL_ALLOWED_USER_IDS` に含まれるDiscordユーザーだけです。
- ComfyUI APIは家PC内のローカルAPIとして使い、インターネットへ直接公開しないでください。
- Botは任意コマンドを実行しません。起動できるのは `.env` で固定した `SD_START_FILE` または `COMFYUI_START_FILE` のみです。
- `SD_BACKEND=comfy` の場合、`/sd_start` は `COMFYUI_START_FILE` の `.bat / .cmd / .exe` だけを起動します。
- Stability Matrix管理のComfyUIを起動する場合は、同梱の `start_comfyui.bat` を使えます。`.env` には `COMFYUI_START_FILE=C:\Users\PC_User\Documents\Codex\2026-06-23\discord-bot-python-discord-py-pc\start_comfyui.bat` を設定してください。
- 起動直後はComfyUI APIがまだ応答しない場合があります。`/sd_start` は数秒待って疎通確認し、未応答なら少し待ってから `/sd_status` を確認するよう案内します。
- 生成枚数、解像度、steps、cfgには上限があります。
- 生成中は二重実行できないようにロックします。
- ComfyUIでは、Workflow JSON内のPositive / Negativeノードを自動検出して書き換えます。
- Seedは `RandomNoise.noise_seed` または `seed` 入力を自動検出し、Bot側で毎回ランダムに設定します。
- `SaveImage` ノードも自動検出し、生成後の画像をDiscordへ返します。

### ComfyUI Workflow設定

ComfyUIのWorkflowは外部JSONとして差し替えできます。

手順:

1. Stability MatrixでComfyUIを起動します。
2. ブラウザで `http://127.0.0.1:8188` が開くことを確認します。
3. ComfyUIで使いたいWorkflowをAPI形式のJSONとして保存します。
4. 保存したJSONを `workflows/comfy_workflow.json` に置きます。
5. `.env` に `COMFY_WORKFLOW_FILE=workflows/comfy_workflow.json` を設定します。
6. `/sd_status` でWorkflowの存在とAPI状態を確認します。

Workflowには以下のノードが必要です。

- Positive用の `CLIPTextEncode`
- Negative用の `CLIPTextEncode`
- `RandomNoise` または `seed` 入力を持つサンプラー
- 完成画像を保存する `SaveImage`

通常はノードID設定は不要です。BotはKSampler系ノードの `positive` / `negative` 接続を辿って `CLIPTextEncode` を検出します。

どうしても自動検出が合わないWorkflowだけ、上書き用にWorkflow JSON内のノードIDを `.env` に設定できます。

```env
COMFY_POSITIVE_NODE_ID=6
COMFY_NEGATIVE_NODE_ID=7
COMFY_SEED_NODE_ID=3
```

### 時間指定シャットダウン

指定した分数後、または指定した時刻にPCをシャットダウンできます。

使い方:
- `/pc_shutdown_in minutes:30`
- `/pc_shutdown_at time:2330`
- `/pc_shutdown_at time:23:30`
- `/pc_shutdown_at time:23時30分`
- `/pc_shutdown_status`
- `/pc_shutdown_cancel`

動作:
- `/pc_shutdown_in minutes:30` は30分後に `shutdown /s /t 1800` を予約します。
- `/pc_shutdown_at` は日本時間で指定時刻までの秒数を計算します。
- 時刻は `23:30`、`2330`、`0730`、`730`、`7:30`、`23時30分`、`23時`、`1時` の形で指定できます。
- 指定時刻が現在時刻より前の場合は、翌日の時刻として扱います。
- 操作パネルの `時間指定シャットダウン` からも、モーダル入力で時刻を指定できます。
- 予約が成功すると、Botは `data/pc_shutdown_state.json` に予約内容を保存します。
- `/pc_shutdown_status` は、この保存内容を表示します。Windows側の予約を直接検出するものではありません。

注意:
- 許可ユーザーだけが使えます。
- 実行前に確認ボタンが出ます。
- 確認ボタンを押すまでシャットダウンは予約されません。
- botが起動しているPCだけを操作できます。
- shutdown予約後は、Windows側に予約されます。
- 取り消す場合は `/pc_shutdown_cancel` を使ってください。
- `/pc_shutdown_cancel` 実行時は、Bot側の予約状態ファイルも削除します。
- PC自体を再起動した場合、Windows側のshutdown予約は消える可能性があります。

### Bot再起動機能

許可ユーザーだけが、DiscordからBot自身を再起動できます。

使い方:

```text
/bot_restart
```

必要な `.env` 設定:

```env
BOT_START_FILE=C:\Users\PC_User\Documents\Codex\2026-06-23\discord-bot-python-discord-py-pc\start_bot.bat
```

注意:

- `PC_CONTROL_ALLOWED_USER_IDS` に含まれるユーザーだけが使えます。
- `BOT_START_FILE` には `.bat / .cmd / .exe` のフルパスだけ指定できます。
- 任意コマンド実行はしません。指定された固定ファイルだけを起動します。
- 実行すると新しいBot起動処理を開始してから、現在のBotプロセスを終了します。

## 状態点検・作業ログ整理

`/health_check` で、Botとローカル環境の状態を手動点検できます。

確認する内容:

- Botの起動時刻、稼働時間、Pythonバージョン、作業ディレクトリ
- `.env` と `.env.example` の存在
- 必須そうな環境変数の未設定警告
- `logs/error.log` と `logs/bot.log` の最終更新時刻、直近エラー件数
- `commands/`、`services/`、`views/`、`README.md`、`.env.example` の直近24時間の更新
- `SD_API_URL` がある場合のStable Diffusion WebUI API疎通
- `LM_STUDIO_API_URL` などがある場合のローカルLLM API疎通

秘密情報について:

- Botトークン、APIキー、secret、password の値はDiscordへ表示しません。
- `.env` の中身そのものは送信せず、設定キーの有無だけを確認します。

使い方:

```text
/health_check
```

このコマンドは `PC_CONTROL_ALLOWED_USER_IDS` に含まれる許可ユーザーだけが使えます。

### 状態点検の定期通知

`.env` に以下を設定すると、指定したチャンネルへ定期的に状態点検結果を送信できます。

```env
HEALTH_CHECK_ENABLED=true
HEALTH_CHECK_INTERVAL_MINUTES=60
HEALTH_CHECK_CHANNEL_ID=通知先チャンネルID
```

注意:

- `HEALTH_CHECK_CHANNEL_ID` が未設定の場合、定期通知は送信されません。
- 疎通確認に失敗してもBot全体は落ちません。警告として表示します。
- 通知が長くなりすぎる場合は一部省略します。
- 直近ログは短く要約し、長文をそのままDiscordへ流さないようにしています。

### LM Studioを点検対象にする場合

LM StudioのOpenAI互換APIを確認したい場合は、`.env` に以下を設定してください。

```env
LM_STUDIO_API_URL=http://127.0.0.1:1234/v1
```

LM Studio側でローカルサーバーが起動していない場合、`/health_check` では接続失敗として表示されます。

## LM Studioプロンプト生成

`/prompt` で、日本語の内容からStable Diffusion向けの `Positive` / `Negative` プロンプトを作れます。
`/generate` では、同じプロンプト生成処理を使って、そのままComfyUIで画像生成まで実行します。
`/generate_raw` では、入力した `Positive` / `Negative` をLM Studioの生成・後処理やMemory追加に通さず、そのままComfyUIへ渡します。
`/generate_ai` では、日本語の要望を `shogou-chan-prompt-designer` 用のプロンプト設計で `positive` / `negative` / `summary` / `assumptions` のJSONに変換し、`positive` / `negative` をそのままComfyUIへ渡します。
`/image_analyze` では、参考画像を画像生成に使えそうな要素へ分解できます。

使い方:

```text
/prompt text:雨の日の小さな喫茶店、窓際、やわらかい光
/generate text:雨の日の小さな喫茶店、窓際、やわらかい光
/generate_raw positive:masterpiece, 1girl, detailed background negative:low quality, blurry
/generate_raw positive:masterpiece, landscape use_default_negative:false
/generate_raw positive:masterpiece, portrait public:false
/generate_ai request:雨の日の小さな喫茶店、窓際、やわらかい光
/generate_ai request:黒髪ロングの女の子 reference_id:1
/image_analyze image:参考画像
/image_analyze image:参考画像 private:true
/image_memory_list
/image_memory_show id:1
/image_memory_delete id:1
```

動き:

1. `fixed_template_llm_instruction.md` をsystem promptとしてLM Studioへ送ります。
2. ユーザーの日本語入力をLM Studioへ送ります。
3. 保存済みメモリがあれば、関連しそうな内容をsystem promptへ短く追記します。
4. LM StudioのJSON返答を `sd_prompt_template_generator.py` の整形・補正ロジックに通します。
5. `/prompt` はDiscordに `Positive` / `Negative` 形式で返します。
6. `/prompt` の返信の `画像生成` ボタンを押すと、表示されたPositive / Negativeをそのまま使ってComfyUI APIへ送信します。
7. `/generate` は確認ボタンなしで、生成されたPositive / NegativeをそのままComfyUI APIへ送信して画像を返します。

`/generate_raw` の動作:

- `positive` は必須、`negative` は省略可能です。各入力は4,000文字までです。
- `negative` を省略すると、既存の `SD_DEFAULT_NEGATIVE_PROMPT` を使います。
- `negative` を省略し、`use_default_negative:false` を指定するとNegativeを空欄にします。
- `negative` を入力した場合は、`use_default_negative` に関係なく入力値を使います。
- `public` の既定値は `true` です。生成成功時の画像・Positive・Negative・画像選別AIの結果は通常のチャンネルメッセージとして投稿され、そのチャンネルを閲覧できる全員に見えます。
- 自分だけに結果を表示したい場合は `public:false` を指定します。この場合だけ、従来どおり生成結果をephemeralで表示します。
- コマンド受付中の表示、権限拒否、入力エラー、生成エラーは `public` に関係なくephemeralです。
- LM Studio、プロンプト後処理、通常Memory、Image Memory、`reference_id` は使いません。
- 生成枚数と生成後の画像選別AIは既存の `/generate` と同じです。
- 生成結果のDiscord表示は長くなりすぎないよう、Positive / Negativeを一部省略する場合があります。ComfyUIへ渡す入力自体は省略しません。

`/generate_ai` の動作:

- 入力項目は `request`（日本語の要望、必須）と `reference_id`（画像参考メモID、任意）だけです。
- system promptには、リポジトリ直下の `shogou_chan_prompt_designer_instruction.md`（`outputs/shogou-chan-prompt-designer/SKILL.md` からClaude Code向けの記述を除いて作成した専用ファイル）を使います。`fixed_template_llm_instruction.md` を使う `/prompt` `/generate` とは別の指示文です。
- 色彩・明るさの判断は、`shogou_chan_prompt_designer_instruction.md` の「色彩・光の設計」セクションで決まります（Phase 1E）。ユーザーの明示指定 → 既知キャラクターの配色 → 髪色・瞳色・衣装 → シーンの種類 → 指定が弱い場合はニュートラルな明るさ、の優先順位で判断し、指定が弱いときの既定は `neutral lighting` / `natural lighting` / `balanced color palette` です。寒色・暗さ・強い逆光をデフォルトで追加しません。
- 通常Memoryと画像参考Memory（Image Memory）はどちらも使います。`reference_id` を指定した場合はそのMemoryのタグを優先し、未指定の場合は既存の自動関連判定をそのまま使います。
- LM Studioの返答は次の順で解析します: `<think>...</think>` の除去 → Markdownコードフェンスの除去 → 前後の説明文からのJSON抽出 → `positive` / `negative` / `summary` / `assumptions` の4キー・型・必須項目の検証。
- `<think>` の閉じタグが無い場合でも、レスポンス全体を誤って消さない実装になっています（閉じタグが見つからない場合は除去処理をスキップし、後続のJSON抽出へそのまま進みます）。
- JSON抽出失敗、必須キー不足、`positive`/`negative`が空、`assumptions`が配列でない、`summary`が文字列でない、のいずれかに該当する場合はComfyUIでの画像生成を開始せず、理由だけを本人に返します。
- `positive` / `negative` は `sd_prompt_template_generator.py` 側のタグ再構成・並び替え・品質タグ付与などのレガシー後処理を一切通さず、LM Studioが返した文字列をそのまま `/generate_raw` と同じ経路でComfyUIへ渡します。
- 生成成功時は `/generate_raw` の既定（`public:true`相当）と同じく通常投稿になり、画像・Summary・Assumptions・Positive・Negative・画像選別AIの結果がチャンネル内の全員に見えます。コマンド受付中の表示、権限拒否、入力エラー、生成エラー、JSON検証エラーは常にephemeralです。
- 生成枚数と生成後の画像選別AIは既存の `/generate` `/generate_raw` と同じです。

`/generate_ai` の生成履歴（Phase 1A・評価収集の土台）:

- 元の依頼（`request`）、生成された `positive`/`negative`/`summary`/`assumptions`、`reference_id`、実際にComfyUIへ送信したワークフローから読み取った生成条件（checkpoint・seed・sampler・scheduler・steps・cfg・width/height・ワークフローファイルとそのハッシュ）、画像選別AIの利用有無と結果を `data/generation_history.db`（SQLite）へ保存します。
- LM StudioのJSON検証成功・ComfyUI生成成功・画像選別AI処理後の3条件がそろったときだけ、1回の `/generate_ai` 呼び出し分をまとめて1トランザクションで保存します。JSON検証失敗・LM Studio呼び出し失敗・ComfyUI生成失敗の場合は履歴を保存しません。画像選別AI自体が使えなかった／失敗した場合は、生成自体は成功しているため「画像選別AI未使用」として保存します。
- 現時点では画像ファイルのコピー・移動は行わず、生成時点のパスをそのまま記録します。そのため元画像が後から削除・移動されると、履歴上のパスは無効になる可能性があります。
- 履歴の保存に失敗しても（DB破損・ディスク容量不足など）、画像生成自体とDiscordへの結果送信は継続されます。失敗はログにのみ記録されます。
- Selection AIの再学習・Prompt Designerの自動改善は未実装です（将来のPhaseで追加予定）。

`/generate_ai` の画像評価ボタン（Phase 1B・Phase 1B-2で3種類に拡張、Phase 1Dで4種類に拡張）:

- `/generate_ai` がDiscordへ送信した画像には、画像ごとに「⭐ お気に入り」「👍 好き」「👎 微妙」「🎯 想定と違う」の4種類のボタンが付きます。ボタンの番号は添付画像の並び順に対応します。
- 「⭐ お気に入り」は、特に完成度が高く今後の理想例・強い正例として使いたい画像に押すボタンです。将来のSelection AI再学習で強い正例として使う想定で、現時点では自動再学習は行いません。
- 「🎯 想定と違う」は画像のクオリティではなく、依頼内容・構図・衣装・キャラクター・雰囲気などが意図から外れた場合の評価です。将来的にPrompt Designerの解釈改善に使う想定で、現時点では自動改善は行いません。
- 押すとその評価が `data/generation_history.db` の `image_ratings` テーブルへ保存され、結果はephemeral（本人にだけ見える表示）で返ります。
- 評価は押した本人のものとして保存されます。別のユーザーが押した場合はそのユーザー自身の評価として別に記録され、同じ人が押し直すと最新の評価へ更新されます（初回評価日時は保持され、更新日時だけが変わります）。
- 画像選別AIによって送信されなかった画像にはボタンが付きません。評価できるのは実際に送信された画像だけです。1画像あたり4ボタンになったうえ、Phase 1Cで再生成ボタン用の行を1つ確保したため、Discordの1メッセージあたりのコンポーネント上限（5行×各5個）内に収めるべく、評価ボタンが付くのは送信順で先頭4枚までです。
- ボタンの有効期限は1時間です。**Bot再起動後のボタン操作には未対応**で、期限切れ・再起動後は押しても反応しません（生成済み画像と保存済み履歴・評価はそのまま残ります）。
- 集めた評価は将来のSelection AI・Prompt Designer改善のための土台で、**現時点では自動再学習や自動改善は一切行いません**。

`/generate_ai` の再生成ボタン（Phase 1C）:

- 評価ボタンの下の行に「🔁 同じ条件で再生成」「🎥 構図を変えて再生成」の2つのボタンが付きます。画像1枚ごとではなく、元の生成リクエスト（generation_requests）単位で1組だけ表示されます。
- 押せるのは、その `/generate_ai` を実行した本人だけです。他のユーザーが押した場合はephemeralで拒否されます。
- 「🔁 同じ条件で再生成」は、保存済みのPositive/NegativeをそのままComfyUIへ渡します。Prompt Designer（LM Studio）は呼び出しません。seedは新しく振り直され、checkpoint・sampler・steps・cfg・width/heightは通常の生成経路（同じComfyUI workflowファイル）を再利用するため、実質的に同条件のまま新しい1枚を作ります。
- 「🎥 構図を変えて再生成」は、元のrequest_text・positive・negative・summary・assumptionsをPrompt Designerへ渡し、「キャラクター・衣装・雰囲気・重要な属性は維持したまま、構図・カメラ角度・ポーズ・画面内配置だけを変更する」よう明示的に指示します。生成後は通常どおり画像選別AIも使われます。
- 再生成結果も新しい `generation_requests` / `generation_images` として `data/generation_history.db` へ保存され、元のrequestとの関係（`parent_request_id`・`regeneration_type`）が記録されます。再生成結果自体にも、評価ボタン・再生成ボタンが同様に付きます。
- 連打対策として、同じ元リクエストに対する再生成はプロセス内で同時に1件までです（Bot再起動をまたいだ永続的な回数制限ではありません）。ボタンを押すとephemeralで「再生成を開始しました。」と返り、失敗時も内部エラーの詳細はDiscordへ出さず、ログにのみ記録します。
- 評価ボタンと同様、**Bot再起動後のボタン操作には未対応**です。

`/generate_ai_stats`（Phase 2A・評価履歴の見える化）:

- 実行者本人の `/generate_ai` 生成履歴と、その画像に対する**本人自身の**評価だけを集計してephemeralなEmbedで表示します（他ユーザーが同じ画像に付けた評価は含みません）。`/generate_ai` と同じ許可ユーザー制限（`is_pc_control_allowed`）を使います。
- 表示内容: 全体件数（リクエスト数・画像数・評価済み/未評価数）、⭐👍👎🎯の評価別件数と好評価率、直近7日/30日/全期間の傾向、通常生成・same_prompt・change_composition別の生成数と評価内訳、保存済みpositive/negative/summary/assumptions/requestから集計したlighting・色温度・明るさ・逆光・室内外・時間帯・構図・主要シーンなどのタグ出現傾向（好評価 favorite/like と低評価 meh/off_target の比較）。
- タグ集計は固定の語句リストを単語境界つきで照合するだけの軽量な文字列集計で、画像解析やPrompt Designerへの自動反映・自動学習は行いません。同一リクエスト内で同じ語句が重複してもリクエストごとに1回だけ数えます。
- 評価済み件数が少ない場合（既定10件未満）や、タグ比較の各グループが小さい場合（既定3件未満）は、断定的な傾向コメントの代わりに「まだ評価数が少ないため、傾向判断には不十分です。」と表示します。
- DB読み取りに失敗した場合もBotは落ちず、ephemeralな短いメッセージを返します。既存の `generation_history.db` のテーブル・カラムは変更していません。

### 画像解析メモ

`/image_analyze image:参考画像` で、添付画像を以下の項目に分解します。

```text
人物
髪色・髪型
目
表情
服装
背景
構図
光・雰囲気
Stable Diffusion向けタグ候補
```

解析結果は既定では通常メッセージとしてチャンネルに残ります。
自分だけに表示したい場合は `private:true` を指定してください。

結果に付く `Memoryに保存` ボタンを押すと、以下の3箇所へ保存します。`name` 未指定時は、保存前に自動名入りの名前入力Modalが開きます。

- `data/memory.json`: 既存のMemory反映用に「画像参考メモ」として保存
- `data/image_memory/ref_0001.json` のような専用JSON: 一覧・詳細表示用に保存
- `data/image_memory/images/ref_0001.webp` のような画像ファイル: 一覧・詳細の元画像表示用に保存

保存した内容は、以後の `/prompt` や `/generate` で関連しそうな情報として使われます。
元画像はIDと対応する安全な固定ファイル名で保存します。古い画像参考メモは元画像なしのまま利用できます。

保存済み画像参考メモの確認:

```text
/image_memory_list
/image_memory_show id:1
/image_memory_rate id:1 rating:5
/image_memory_use id:1 enabled:true
/image_memory_delete id:1
```

専用JSONには以下を保存します。

```text
id, user_id, name, title, created_at, note, analysis_text, image_path,
rating, use_for_generation,
person, hair, eyes, expression, outfit, background,
composition, lighting_mood, sd_tags, memory_id
```

一覧が長い場合は `page` を指定できます。

```text
/image_memory_list page:2
```

必要な `.env` 設定:

```env
LMSTUDIO_API_URL=http://127.0.0.1:1234/v1
LMSTUDIO_TEXT_MODEL=テキスト用モデルID
LMSTUDIO_VISION_MODEL=Vision用モデルID

# 後方互換用のフォールバックです。
# 上の2つが未設定の場合に使います。
LMSTUDIO_MODEL=local-model
```

LM Studio側の確認:

- LM StudioでLocal Serverを起動する
- OpenAI互換APIが使える状態にする
- `LMSTUDIO_API_URL` は通常 `http://127.0.0.1:1234/v1` を指定する
- `/chat`、`/prompt`、`/generate` は `LMSTUDIO_TEXT_MODEL` を使います。
- `/image_analyze` は `LMSTUDIO_VISION_MODEL` を使います。
- どちらも未設定の場合は、後方互換用に `LMSTUDIO_MODEL` を使います。
- `/image_analyze` を使う場合は、画像入力に対応したVisionモデルをLM Studioで読み込む

注意:

- LM Studioが起動していない場合、`/prompt` は接続失敗を返します。
- `/generate` もLM StudioとComfyUI APIの両方が起動している必要があります。
- `/image_analyze` はLM Studioのモデルが画像入力に対応していない場合、解析に失敗します。
- `画像生成` ボタンを使うには、`SD_API_URL` のComfyUI APIが起動している必要があります。
- `画像生成` ボタンは、`/prompt` を実行した本人だけが押せます。
- 画像生成は、`PC_CONTROL_ALLOWED_USER_IDS` に含まれる許可ユーザーだけが実行できます。
- `/prompt` では安全判定は無効化済みです。LM Studioの `safe_check`、`blocked_elements`、`adult`、`consent_ok` による拒否は行わず、Positive/Negative整形を続行します。
- 返答はDiscordに長く出すぎないよう、整形して返します。

### /prompt の固定整形ルール

`/prompt` の出力品質を安定させるため、LM Studioの返答後に `sd_prompt_template_generator.py` で必ず補正します。

Positiveの先頭には、必ず以下を付けます。

```text
masterpiece, best quality, anime style, 1girl, solo
```

NegativeはLLMには生成させず、以下の固定値を使います。

```text
worst quality, low quality, normal quality, bad anatomy, bad hands, extra fingers, missing fingers, extra limbs, deformed, blurry, text, watermark, logo, jpeg artifacts, bad eyes, distorted eyes, asymmetrical eyes, glowing eyes, overly bright eyes, two-tone hair, multicolored hair, face paint, cheek mark, facial mark, leaf mark, asymmetrical hair color
```

以下の語はNegativeには入れません。

```text
cartoonish style, unrealistic proportions, poor lighting, dark tones
```

また、`short haircut girl` のような自然文タグは、できるだけ `short hair, 1girl` のようなタグ列に正規化します。

日本語でよく使う要素は、入力文にその語がある場合だけ Danbooru 風タグへ展開します。
重複したタグは自動で削除されます。

```text
サンタ服 -> santa hat, red santa dress, white fur trim, christmas costume, christmas lights, snow, gift box
巫女 -> miko, red hakama, white kimono, shrine maiden
制服 -> school uniform, sailor uniform, pleated skirt, ribbon
メイド -> maid, maid headdress, apron, frills
カフェ -> coffee shop, cafe interior, wooden table, coffee cup, counter, dessert, cozy atmosphere, warm interior
神社 -> shrine, torii, stone pathway, japanese architecture
雨 -> rain, wet ground, umbrella, cloudy sky
```

入力文に該当語がある場合だけ、Negativeにも補助タグを追加します。

```text
カフェ -> classroom, school desk, blackboard
サンタ服 -> face paint, cheek mark, leaf mark, multicolored hair, two-tone hair
```

保存済みメモリ由来の背景・服装タグは、今回の入力に別の背景や服装がある場合は混ざりすぎないように抑えています。
`サンタ服` のときは `red and green colors` のような抽象的な色タグを避け、髪色が入力で指定されていない場合は髪色タグを勝手に混ぜないようにしています。

### /prompt の Animagine XL 4.0 向けタグ整形

`/prompt` の Positive は、Stable Diffusion / ComfyUI に貼りやすい Danbooru 風タグへ整形します。

Positive の先頭には、品質タグとして以下を固定で入れます。

```text
masterpiece, best quality, newest, very aesthetic, absurdres, anime style, 1girl, solo
```

その後ろのタグは、以下の順に並べます。

1. 品質タグ
2. 人物タグ
3. 髪・目
4. 表情・視線
5. ポーズ
6. 服装
7. 背景
8. 光・雰囲気

`cute 1girl` のような複合自然文タグは、`cute, 1girl` のように分解します。
`cool colors` と `warm lighting` のような矛盾しやすいタグが同時に出た場合は、より先に出た指定を優先して整理します。
保存済みメモリ由来のタグと今回の入力が矛盾した場合も、今回の入力に近いタグが残りやすいようにしています。

Negative はこれまで通り固定値を使い、LLMには生成させません。

人物の品質を安定させるため、Positive には以下の人物・顔・目・構図タグを標準で追加します。

```text
detailed face, beautiful face, detailed eyes, expressive eyes, upper body, looking at viewer, centered composition
```

`kimono` や和服系タグが含まれる場合は、衣装の描き込みを強めるために以下も補います。

```text
floral kimono, detailed kimono, long sleeves, obi
```

背景タグよりも人物タグが前に並ぶようにし、夕焼けや暖色系の入力がある場合は、メモリ由来で混ざった `cool colors` / `blue tones` を除外します。

## 画像生成後の画像選別AI連携

初号ちゃんで画像生成したあと、既存の `local_ai_project` にある画像選別AIで、生成直後の画像だけを選別できます。

`.env` に以下を設定してください。

```env
LOCAL_AI_PROJECT_DIR=C:\Users\PC_User\Documents\Codex\2026-07-04\ai-phase-1-comfyui-outputs-raw\local_ai_project
SD_SELECT_GENERATE_COUNT=5
SD_SELECT_RETURN_COUNT=1
```

動作:

1. `/prompt` の「画像生成」ボタン、`/generate`、`/generate_raw`、`/generate_ai`、または `/sd_generate` で画像を生成します。
2. 初号ちゃんは今回生成された画像だけを一時フォルダへコピーします。
3. `LOCAL_AI_PROJECT_DIR\scripts\score_batch_top5.py` にその一時フォルダを渡します。
4. 選別AIが `outputs\auto_selected\latest_top5` に出力した上位画像をDiscordへ返します。

`/prompt` の「画像生成」ボタン、`/generate`、`/generate_raw`、`/generate_ai` では、画像選別AIが使える場合だけ複数枚生成します。

- `SD_SELECT_GENERATE_COUNT`: 選別AI用に生成する枚数です。5〜20の範囲で指定します。未設定時は5枚です。
- `SD_SELECT_RETURN_COUNT`: 選別後にDiscordへ返す枚数です。1〜3の範囲で指定します。未設定時は1枚です。

画像選別AIが未設定の場合や、1枚だけ生成した場合は、選別AIの上位表示ではなく通常の「画像を生成しました」という表示になります。

注意:

- `LOCAL_AI_PROJECT_DIR` が未設定の場合は、画像選別AIをスキップして通常の生成画像を返します。
- `score_batch_top5.py` の実行に失敗した場合も、通常の生成画像を返します。
- 既存の生成画像保存先 `data\sd_outputs` は変更しません。
- 選別AI側の `model\ranker.pkl` や依存ライブラリが必要です。

### キャラクター辞書（キャラクター名・作品名の解決）

`services/character_tag_dictionary.py` は、要望文に含まれる既知キャラクターの日本語名・短縮名・英語名・作品名付き表記などを、Prompt Designerが使う正式な character tag / series tag へ解決する辞書です。解決できたキャラクターは `Known character:` という補助情報としてPrompt Designerのsystem promptへ渡され、髪色・衣装などの外見は推測・追加しません（外見の補強は別機能です）。1回の要望に複数キャラクターが含まれる場合は、勝手に1人へ絞り込まず複数キャラクターとして渡します。辞書内の1エントリが壊れていても、そのエントリだけがログ付きでスキップされ、Bot全体は起動を継続します。

キャラクターの追加方法・エイリアスの考え方（`safe_aliases` / `ambiguous_aliases` の使い分け）は `services/character_tag_dictionary.py` 冒頭のモジュールdocstringにまとまっています。

### `/image_analyze` の構造化タグ

Visionモデルの解析結果は、人物・髪・目・表情・服装・背景・構図・光/雰囲気・SDタグ候補のJSONとして受け取り、英語のDanbooru風タグへ正規化します。コードフェンス付きJSONにも対応し、JSONが壊れている場合は従来の自由文解析へフォールバックします。

保存先の `data/image_memory/ref_XXXX.json` では各カテゴリを配列で保持し、`analysis_text` は人間向けの要約として残します。Discordでは「日本語の要約」と「SDタグ候補」を分けて表示します。
