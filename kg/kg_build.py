import pandas as pd
from neo4j import GraphDatabase
import re

# ===================== 需要修改的部分 START =====================
NEO4J_URI = "bolt://localhost:7687"      # Neo4j地址
NEO4J_USER = "neo4j"                           # 用户名
NEO4J_PASSWORD = "12345678"        # 密码
FILE1 = r"G:\计算机设计大赛\data\中草药药效汇总表.xls"   # 第一个表
FILE2 = r"G:\计算机设计大赛\data\中药功效数据_注意事项.xlsx"  # 第二个表
# ===================== 需要修改的部分 END =====================


class TCMKnowledgeGraph:
    from neo4j import GraphDatabase
    from neo4j.exceptions import ServiceUnavailable, Neo4jError

    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(
            uri,
            auth=(user, password),
            max_connection_lifetime=30 * 60,  # 30分钟
            connection_timeout=30,  # 30秒连接超时
            keep_alive=True,  # 保持连接活跃
            max_connection_pool_size=50  # 连接池大小
        )

    def connect(self):
        """测试连接"""
        try:
            with self.driver.session() as session:
                result = session.run("RETURN 'Connected' as status")
                print(f"✓ Neo4j连接成功: {result.single()['status']}")
                return True
        except Exception as e:
            print(f"✗ Neo4j连接失败: {e}")
            print("请检查: 1.Neo4j是否启动 2.连接参数是否正确")
            return False

    def load_and_merge_data(self):
        # 读取第一个表（汇总表），跳过第一行标题，只读取C到K列（索引列号2到10）
        df1 = pd.read_excel(FILE1, header=1, usecols="C:K")
        # 手动指定列名（确保顺序正确）
        df1.columns = ['类别', '名称', '别名', '性味', '归经', '功效', '适应症', '用量', '注意事项']
        df1 = df1.dropna(how='all')
        print(f"✓ 第一个表加载: {len(df1)} 条记录")

        # 读取第二个表（功效数据）
        df2 = pd.read_excel(FILE2)
        df2.columns = ['名称', '功效', '注意事项']  # 重命名列
        df2 = df2.dropna(how='all')
        print(f"✓ 第二个表加载: {len(df2)} 条记录")

        # 为df2添加缺失的列，并设为空字符串
        for col in ['类别', '别名', '性味', '归经', '适应症', '用量']:
            df2[col] = ''

        # 合并两个DataFrame
        df_combined = pd.concat([df1, df2], ignore_index=True, sort=False)

        # 按“名称”分组，合并相同中药的信息
        def combine_series(series):
            """取第一个非空值，用于非文本字段"""
            non_null = series.dropna()
            return non_null.iloc[0] if not non_null.empty else ''

        def combine_text(series, sep='; '):
            """合并文本字段，去重后拼接"""
            texts = [str(x).strip() for x in series.dropna() if str(x).strip()]
            # 简单去重（按整个字符串）
            unique = []
            for t in texts:
                if t not in unique:
                    unique.append(t)
            return sep.join(unique)

        # 分组聚合
        df_merged = df_combined.groupby('名称', as_index=False).agg({
            '类别': combine_series,
            '别名': combine_series,
            '性味': combine_series,
            '归经': combine_series,
            '用量': combine_series,
            '适应症': combine_series,
            '功效': lambda x: combine_text(x, '；'),
            '注意事项': lambda x: combine_text(x, '；')
        })

        print(f"✓ 合并后中药总数: {len(df_merged)} 条")
        self.df = df_merged
        return True

    def clear_database(self):
        """清空现有数据（可选）"""
        confirm = input("是否清空现有数据？(y/n): ").strip().lower()
        if confirm == 'y':
            with self.driver.session() as session:
                session.run("MATCH (n) DETACH DELETE n")
                print("✓ 数据库已清空")

    def _split_text(self, text):
        """将文本按常见分隔符拆分为列表，去除空项"""
        if pd.isna(text) or not text:
            return []
        # 替换中文逗号、顿号为英文逗号，然后分割
        text = str(text).replace('，', ',').replace('、', ',').replace(' ', ',')
        items = [item.strip() for item in text.split(',') if item.strip()]
        return items

    def create_nodes_and_relationships(self):
        """创建节点和关系"""
        with self.driver.session() as session:
            # 创建索引
            session.run("CREATE INDEX IF NOT EXISTS FOR (m:Medicine) ON (m.name)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (e:Efficacy) ON (e.name)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (d:Disease) ON (d.name)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (c:Category) ON (c.name)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (m:Meridian) ON (m.name)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (t:TasteProperty) ON (t.name)")

            count = 0
            for _, row in self.df.iterrows():
                name = str(row['名称']).strip() if pd.notna(row['名称']) else ''
                if not name:
                    continue

                alias = str(row['别名']).strip() if pd.notna(row['别名']) else ''
                taste = str(row['性味']).strip() if pd.notna(row['性味']) else ''
                meridian = str(row['归经']).strip() if pd.notna(row['归经']) else ''
                dosage = str(row['用量']).strip() if pd.notna(row['用量']) else ''
                precaution = str(row['注意事项']).strip() if pd.notna(row['注意事项']) else ''
                category = str(row['类别']).strip() if pd.notna(row['类别']) else ''

                # 创建中药节点
                session.run("""
                    MERGE (m:Medicine {name: $name})
                    SET m.alias = $alias,
                        m.taste = $taste,
                        m.meridian = $meridian,
                        m.dosage = $dosage,
                        m.precaution = $precaution,
                        m.category = $category
                """, name=name, alias=alias, taste=taste, meridian=meridian,
                     dosage=dosage, precaution=precaution, category=category)

                # 类别节点
                if category:
                    session.run("""
                        MERGE (c:Category {name: $cat})
                        WITH c
                        MATCH (m:Medicine {name: $name})
                        MERGE (m)-[:BELONGS_TO]->(c)
                    """, cat=category, name=name)

                # 归经节点
                meridians = self._split_text(meridian)
                for m in meridians:
                    session.run("""
                        MERGE (mer:Meridian {name: $m})
                        WITH mer
                        MATCH (med:Medicine {name: $name})
                        MERGE (med)-[:HAS_MERIDIAN]->(mer)
                    """, m=m, name=name)

                # 性味节点
                tastes = self._split_text(taste)
                for t in tastes:
                    session.run("""
                        MERGE (tp:TasteProperty {name: $t})
                        WITH tp
                        MATCH (med:Medicine {name: $name})
                        MERGE (med)-[:HAS_TASTE]->(tp)
                    """, t=t, name=name)

                # 功效节点
                efficacy_desc = str(row['功效']).strip() if pd.notna(row['功效']) else ''
                if efficacy_desc:
                    efficacy_items = self._split_text(efficacy_desc)
                    for eff in efficacy_items:
                        session.run("""
                            MERGE (e:Efficacy {name: $eff})
                            WITH e
                            MATCH (m:Medicine {name: $name})
                            MERGE (m)-[:HAS_EFFICACY {description: $full}]->(e)
                        """, eff=eff, full=efficacy_desc, name=name)

                # 适应症（疾病）节点
                indication_desc = str(row['适应症']).strip() if pd.notna(row['适应症']) else ''
                if indication_desc:
                    diseases = self._split_text(indication_desc)
                    for dis in diseases:
                        session.run("""
                            MERGE (d:Disease {name: $dis})
                            WITH d
                            MATCH (m:Medicine {name: $name})
                            MERGE (m)-[:TREATS {description: $full}]->(d)
                        """, dis=dis, full=indication_desc, name=name)

                count += 1
                if count % 50 == 0:
                    print(f"已处理 {count} 条记录...")

            print(f"✓ 完成创建: {count} 条中药记录")

    def show_statistics(self):
        """显示统计信息"""
        with self.driver.session() as session:
            print("\n=== 知识图谱统计 ===")
            result = session.run("""
                MATCH (n)
                RETURN labels(n)[0] as type, count(*) as count
                ORDER BY count DESC
            """)
            for record in result:
                print(f"{record['type']}: {record['count']}个")

            print("\n=== 关系统计 ===")
            result = session.run("""
                MATCH ()-[r]->()
                RETURN type(r) as relationship, count(*) as count
                ORDER BY count DESC
            """)
            for record in result:
                print(f"{record['relationship']}: {record['count']}条")

    def run_sample_queries(self):
        """交互式查询"""
        print("\n" + "=" * 50)
        print("进入交互式查询模式")
        print("=" * 50)

        while True:
            print("\n--- 请选择查询类型 ---")
            print("1. 按功效关键词查询（如：清热、解毒）")
            print("2. 按治疗疾病查询（如：咳嗽、头痛）")
            print("3. 查询有注意事项的中药")
            print("4. 查询特定中药的详细信息")
            print("5. 按类别查询中药")
            print("6. 按归经查询中药")
            print("0. 退出查询")

            choice = input("\n请输入选项编号 (0-6): ").strip()

            if choice == '0':
                print("退出查询模式。")
                break
            elif choice == '1':
                keyword = input("请输入功效关键词: ").strip()
                if keyword:
                    self._query_by_efficacy(keyword)
            elif choice == '2':
                disease = input("请输入疾病名称: ").strip()
                if disease:
                    self._query_by_disease(disease)
            elif choice == '3':
                self._query_with_precautions()
            elif choice == '4':
                med_name = input("请输入中药名: ").strip()
                if med_name:
                    self._query_medicine_detail(med_name)
            elif choice == '5':
                cat = input("请输入类别（如解表药）: ").strip()
                if cat:
                    self._query_by_category(cat)
            elif choice == '6':
                mer = input("请输入归经（如肝、肺）: ").strip()
                if mer:
                    self._query_by_meridian(mer)
            else:
                print("输入无效，请重新输入。")

    def _query_by_efficacy(self, keyword):
        with self.driver.session() as session:
            print(f"\n查询结果【功效包含 '{keyword}' 的中药】:")
            result = session.run("""
                MATCH (m:Medicine)-[:HAS_EFFICACY]->(e:Efficacy)
                WHERE e.name CONTAINS $keyword
                RETURN DISTINCT m.name as 中药, e.name as 功效
                LIMIT 20
            """, keyword=keyword)
            count = 0
            for record in result:
                count += 1
                print(f"  {count}. {record['中药']} → {record['功效']}")
            if count == 0:
                print("  （未找到相关结果）")

    def _query_by_disease(self, disease):
        with self.driver.session() as session:
            print(f"\n查询结果【可治疗 '{disease}' 的中药】:")
            result = session.run("""
                MATCH (m:Medicine)-[:TREATS]->(d:Disease)
                WHERE d.name CONTAINS $disease
                RETURN m.name as 中药, d.name as 疾病
                LIMIT 20
            """, disease=disease)
            count = 0
            for record in result:
                count += 1
                print(f"  {count}. {record['中药']} → {record['疾病']}")
            if count == 0:
                print("  （未找到相关结果）")

    def _query_with_precautions(self):
        with self.driver.session() as session:
            print(f"\n查询结果【有注意事项的中药】:")
            result = session.run("""
                MATCH (m:Medicine)
                WHERE m.precaution IS NOT NULL AND m.precaution <> ''
                RETURN m.name as 中药, m.precaution as 注意事项
                LIMIT 20
            """)
            count = 0
            for record in result:
                count += 1
                print(f"  {count}. {record['中药']}: {record['注意事项'][:60]}...")
            if count == 0:
                print("  （未找到有注意事项的中药）")

    def _query_medicine_detail(self, name):
        with self.driver.session() as session:
            print(f"\n查询结果【中药 '{name}' 的详细信息】:")
            result = session.run("""
                MATCH (m:Medicine {name: $name})
                RETURN m
            """, name=name)
            record = result.single()
            if not record:
                print(f"  未找到名为 '{name}' 的中药")
                return
            props = record['m']
            print(f"  中药名: {props.get('name', '')}")
            print(f"  别名: {props.get('alias', '')}")
            print(f"  性味: {props.get('taste', '')}")
            print(f"  归经: {props.get('meridian', '')}")
            print(f"  用量: {props.get('dosage', '')}")
            print(f"  注意事项: {props.get('precaution', '')}")
            print(f"  类别: {props.get('category', '')}")

            print(f"\n  功效:")
            result = session.run("""
                MATCH (m:Medicine {name: $name})-[r:HAS_EFFICACY]->(e:Efficacy)
                RETURN e.name as 功效, r.description as 完整描述
            """, name=name)
            for i, rec in enumerate(result, 1):
                desc = rec['完整描述'] if rec['完整描述'] else rec['功效']
                print(f"    {i}. {desc}")

            print(f"\n  适应症:")
            result = session.run("""
                MATCH (m:Medicine {name: $name})-[r:TREATS]->(d:Disease)
                RETURN d.name as 疾病, r.description as 完整描述
            """, name=name)
            for i, rec in enumerate(result, 1):
                desc = rec['完整描述'] if rec['完整描述'] else rec['疾病']
                print(f"    {i}. {desc}")

    def _query_by_category(self, category):
        with self.driver.session() as session:
            print(f"\n查询结果【类别为 '{category}' 的中药】:")
            result = session.run("""
                MATCH (m:Medicine)-[:BELONGS_TO]->(c:Category {name: $cat})
                RETURN m.name as 中药
                LIMIT 20
            """, cat=category)
            count = 0
            for record in result:
                count += 1
                print(f"  {count}. {record['中药']}")
            if count == 0:
                print("  （未找到相关结果）")

    def _query_by_meridian(self, meridian):
        with self.driver.session() as session:
            print(f"\n查询结果【归经为 '{meridian}' 的中药】:")
            result = session.run("""
                MATCH (m:Medicine)-[:HAS_MERIDIAN]->(mer:Meridian {name: $mer})
                RETURN m.name as 中药
                LIMIT 20
            """, mer=meridian)
            count = 0
            for record in result:
                count += 1
                print(f"  {count}. {record['中药']}")
            if count == 0:
                print("  （未找到相关结果）")


def main():
    print("=" * 50)
    print("中药知识图谱生成器（双文件合并版）")
    print("=" * 50)

    kg = TCMKnowledgeGraph(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

    if not kg.connect():
        return

    if not kg.load_and_merge_data():
        return

    kg.clear_database()
    kg.create_nodes_and_relationships()
    kg.show_statistics()
    kg.run_sample_queries()

    print("\n" + "=" * 50)
    print("完成！在Neo4j Browser中可运行以下示例查询：")
    print("=" * 50)
    print("\n1. 查看所有中药:")
    print("   MATCH (m:Medicine) RETURN m.name, m.category LIMIT 20")
    print("\n2. 查看中药及其功效:")
    print("   MATCH (m:Medicine)-[:HAS_EFFICACY]->(e:Efficacy) RETURN m.name, e.name LIMIT 20")
    print("\n3. 查看治疗特定疾病的中药:")
    print("   MATCH (m:Medicine)-[:TREATS]->(d:Disease {name: '咳嗽'}) RETURN m.name")
    print("\n4. 查看某类别的所有中药:")
    print("   MATCH (m:Medicine)-[:BELONGS_TO]->(:Category {name: '解表药'}) RETURN m.name")
    print("\n5. 可视化查询:")
    print("   MATCH path = (m:Medicine)-[*1..2]-(n) RETURN path LIMIT 50")


if __name__ == "__main__":
    main()