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

---

## 用 Docker Compose 启动（可选）

如果你有 Docker，也可以使用仓库根目录的 `docker-compose.yml` 启动 Neo4j：

1. 复制根目录 `.env.example` 为 `.env`，设置 `NEO4J_USER` / `NEO4J_PASSWORD`（或使用 `NEO4J_USERNAME`）
2. 启动：

```bash
docker compose up -d
```

然后访问：
- Neo4j Browser：`http://localhost:7474`
- Bolt：`bolt://localhost:7687`
