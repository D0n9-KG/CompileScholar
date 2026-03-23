import { useI18n } from '../i18n'

export default function SchemaPage() {
  const { t } = useI18n()

  return (
    <div className="stack">
      <section className="panel">
        <div className="panelHeader">
          <div className="split">
            <div className="panelTitle">{t('PaperLogicTrace 编译策略', 'PaperLogicTrace Compiler Policy')}</div>
            <span className="pill">{t('迁移中', 'Migration')}</span>
          </div>
        </div>
        <div className="panelBody">
          <div className="metaLine">
            {t(
              '旧版第二层前端编辑器已经退场，新的配置界面将围绕 PaperLogicTrace、ResearchMove 和 EvidenceAnchor 重新构建。',
              'The previous second-layer editor has been retired. The next configuration surface is being rebuilt around PaperLogicTrace, ResearchMove, and EvidenceAnchor.',
            )}
          </div>
        </div>
      </section>

      <section className="panel">
        <div className="panelHeader">
          <div className="panelTitle">{t('当前保留的能力', 'What Remains Available Now')}</div>
        </div>
        <div className="panelBody">
          <div className="list">
            <div className="itemCard">
              <div className="itemTitle">{t('版本切换', 'Version Switching')}</div>
              <div className="hint">
                {t(
                  '上方仍可切换不同论文类型使用的编译策略版本，影响后续导入、替换与重建。',
                  'Use the switcher above to activate compiler-policy versions per paper type for future ingest, replace, and rebuild tasks.',
                )}
              </div>
            </div>
            <div className="itemCard">
              <div className="itemTitle">{t('并发与提供方配置', 'Runtime and Provider Controls')}</div>
              <div className="hint">
                {t(
                  '右侧配置中心仍然保留运行并发、嵌入服务和 LLM worker 路由等可执行配置。',
                  'Runtime concurrency, embedding providers, and LLM worker routing remain configurable in the surrounding Config Center.',
                )}
              </div>
            </div>
            <div className="itemCard">
              <div className="itemTitle">{t('下一版目标', 'Next Editor Scope')}</div>
              <div className="hint">
                {t(
                  '新版编辑器会直接面向 ResearchMove 抽取、EvidenceAnchor 证据绑定，以及 derived views 的编译预算控制。',
                  'The next editor will target ResearchMove extraction, EvidenceAnchor binding, and derived-view compiler budget controls.',
                )}
              </div>
            </div>
          </div>
        </div>
      </section>
    </div>
  )
}
