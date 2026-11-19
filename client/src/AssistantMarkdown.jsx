import { memo } from 'react'
import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

const plugins = [remarkGfm]
const components = {
  a: ({ children }) => <span>{children}</span>,
  img: ({ alt }) => <>{alt}</>,
  table: ({ children }) => (
    <div className="markdown-table" role="region" aria-label="Table in assistant reply" tabIndex={0}>
      <table>{children}</table>
    </div>
  ),
  th: ({ children, style }) => <th scope="col" style={style}>{children}</th>,
}

export default memo(function AssistantMarkdown({ content }) {
  return (
    <div className="assistant-markdown">
      <Markdown skipHtml remarkPlugins={plugins} components={components}>{content}</Markdown>
    </div>
  )
})
