# Neo4j 本地启动（无 Docker）

如果你没有 Docker，推荐使用 Neo4j Desktop 在本机启动 Neo4j 5.x。

1. 安装 Neo4j Desktop
2. 新建一个 DBMS（Neo4j 5.x）
3. 设置用户名/密码（默认用户名为 `neo4j`）
4. 启动 DBMS，并确认端口：
   - Bolt：`7687`
   - Browser：`7474`

在项目根目录的 `.env`（或 `backend/.env`）配置：

```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=neo4j_password
```

启动后端后：
- 先调用 `POST /ingest/path` 导入 MinerU 输出的 `*.md`
- 再打开 Neo4j Browser 验证图结构（`Paper` / `Chunk` / `ReferenceEntry` / `CITES` 等）
