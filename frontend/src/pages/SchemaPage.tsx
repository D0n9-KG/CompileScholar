import { useEffect, useMemo } from 'react'

import { useI18n } from '../i18n'

type SchemaPageProps = {
  jumpTarget?: string | null
  jumpFocusKey?: string | null
  jumpNonce?: number
}

type JumpArea = 'rules' | 'prompts' | ''

function resolveJumpArea(target: string | null | undefined): JumpArea {
  if (target === 'schema.rules_json') return 'rules'
  if (target === 'schema.prompts_json') return 'prompts'
  return ''
}

function highlightStyle(active: boolean) {
  return active
    ? {
        borderColor: 'rgba(74, 123, 255, 0.42)',
        boxShadow: '0 0 0 1px rgba(74, 123, 255, 0.2), 0 18px 36px rgba(74, 123, 255, 0.14)',
        animation: 'schema-jump-flash 1.8s ease-out',
      }
    : undefined
}

export default function SchemaPage({ jumpTarget, jumpFocusKey, jumpNonce }: SchemaPageProps = {}) {
  const { t } = useI18n()
  const jumpArea = useMemo(() => resolveJumpArea(jumpTarget), [jumpTarget])

  useEffect(() => {
    if (!jumpArea) return
    const targetId = jumpArea === 'rules' ? 'schema-rules-json' : 'schema-prompts-json'
    document.getElementById(targetId)?.scrollIntoView({ behavior: 'smooth', block: 'center' })
  }, [jumpArea, jumpNonce])

  const jumpMessage = useMemo(() => {
    if (!jumpArea) return ''
    if (jumpArea === 'rules') {
      return t(
        '你刚才跳转的是旧版 rules_json 编辑区。该编辑器已经退役，这里保留的是迁移说明和后续编译器方向。',
        'You jumped to the retired rules_json editor area. This page now keeps migration guidance and the next compiler direction instead.',
      )
    }
    return t(
      '你刚才跳转的是旧版 prompts_json 编辑区。该编辑器已经退役，这里保留的是迁移说明和后续编译器方向。',
      'You jumped to the retired prompts_json editor area. This page now keeps migration guidance and the next compiler direction instead.',
    )
  }, [jumpArea, t])

  const jumpKeyMessage = jumpFocusKey?.trim()
    ? t(`原建议 key: ${jumpFocusKey}`, `Suggested key from the previous flow: ${jumpFocusKey}`)
    : ''

  return (
    <div className="stack">
      <style>{`
        @keyframes schema-jump-flash {
          0% {
            border-color: rgba(74, 123, 255, 0.42);
            box-shadow: 0 0 0 1px rgba(74, 123, 255, 0.2), 0 18px 36px rgba(74, 123, 255, 0.14);
          }
          100% {
            border-color: rgba(15, 23, 42, 0.08);
            box-shadow: none;
          }
        }
      `}</style>

      {jumpArea ? (
        <section className="panel">
          <div className="panelHeader">
            <div className="split">
              <div className="panelTitle">{t('迁移提示', 'Migration Note')}</div>
              <span className="pill">{jumpArea === 'rules' ? 'rules_json' : 'prompts_json'}</span>
            </div>
          </div>
          <div className="panelBody" aria-live="polite">
            <div className="metaLine">{jumpMessage}</div>
            {jumpKeyMessage ? <div className="hint" style={{ marginTop: 8 }}>{jumpKeyMessage}</div> : null}
          </div>
        </section>
      ) : null}

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

      <section
        key={jumpArea === 'rules' ? `schema-rules-${jumpNonce ?? 0}` : 'schema-rules-idle'}
        id="schema-rules-json"
        className="panel"
        style={highlightStyle(jumpArea === 'rules')}
      >
        <div className="panelHeader">
          <div className="panelTitle">{t('rules_json 迁移说明', 'rules_json Migration')}</div>
        </div>
        <div className="panelBody">
          <div className="metaLine">
            {t(
              '旧版规则 JSON 编辑器已下线。后续会拆分为更细粒度的编译器开关，而不是继续维护一块大 JSON 面板。',
              'The legacy rules JSON editor has been retired. Future controls will be exposed as smaller compiler-specific switches instead of one large JSON editor.',
            )}
          </div>
        </div>
      </section>

      <section
        key={jumpArea === 'prompts' ? `schema-prompts-${jumpNonce ?? 0}` : 'schema-prompts-idle'}
        id="schema-prompts-json"
        className="panel"
        style={highlightStyle(jumpArea === 'prompts')}
      >
        <div className="panelHeader">
          <div className="panelTitle">{t('prompts_json 迁移说明', 'prompts_json Migration')}</div>
        </div>
        <div className="panelBody">
          <div className="metaLine">
            {t(
              '旧版 prompts JSON 编辑器已下线。后续提示词能力会并入新的 PaperLogicTrace 编译器配置流，而不是停留在旧的第二层编辑器里。',
              'The legacy prompts JSON editor has been retired. Prompt controls will move into the new PaperLogicTrace compiler workflow instead of the old second-layer editor.',
            )}
          </div>
        </div>
      </section>
    </div>
  )
}
