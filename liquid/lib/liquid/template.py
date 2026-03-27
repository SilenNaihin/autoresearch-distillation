"""
Minimal Liquid-style template engine.
Modify this file to reduce combined parse+render time.
"""

import re


class Template:
    """Simple template engine with variable substitution and basic for loops."""

    def __init__(self, source):
        self.source = source
        self.ast = None

    def parse(self):
        """Parse template source into an AST (list of nodes)."""
        tokens = re.split(r'(\{\{.*?\}\}|\{%.*?%\})', self.source)
        self.ast = []
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if token.startswith('{{') and token.endswith('}}'):
                var_name = token[2:-2].strip()
                self.ast.append(('var', var_name))
            elif token.startswith('{%') and token.endswith('%}'):
                tag_content = token[2:-2].strip()
                if tag_content.startswith('for '):
                    parts = tag_content.split()
                    # {% for item in collection %}
                    loop_var = parts[1]
                    collection_var = parts[3]
                    self.ast.append(('for_start', loop_var, collection_var))
                elif tag_content == 'endfor':
                    self.ast.append(('for_end',))
                elif tag_content.startswith('if '):
                    condition = tag_content[3:].strip()
                    self.ast.append(('if_start', condition))
                elif tag_content == 'endif':
                    self.ast.append(('if_end',))
                else:
                    self.ast.append(('text', token))
            else:
                if token:
                    self.ast.append(('text', token))
            i += 1
        return self

    def render(self, context):
        """Render the parsed template with the given context dict."""
        if self.ast is None:
            self.parse()
        return self._render_nodes(self.ast, context)

    def _render_nodes(self, nodes, context):
        output = []
        i = 0
        while i < len(nodes):
            node = nodes[i]
            if node[0] == 'text':
                output.append(node[1])
            elif node[0] == 'var':
                val = self._resolve(node[1], context)
                output.append(str(val) if val is not None else '')
            elif node[0] == 'for_start':
                loop_var, collection_var = node[1], node[2]
                # Find matching endfor
                body_nodes, end_idx = self._find_block(nodes, i, 'for_start', 'for_end')
                collection = self._resolve(collection_var, context)
                if collection:
                    for item in collection:
                        child_ctx = dict(context)
                        child_ctx[loop_var] = item
                        output.append(self._render_nodes(body_nodes, child_ctx))
                i = end_idx
            elif node[0] == 'if_start':
                condition = node[1]
                body_nodes, end_idx = self._find_block(nodes, i, 'if_start', 'if_end')
                if self._resolve(condition, context):
                    output.append(self._render_nodes(body_nodes, context))
                i = end_idx
            i += 1
        return ''.join(output)

    def _resolve(self, name, context):
        """Resolve a dotted variable name from context."""
        parts = name.split('.')
        val = context
        for part in parts:
            if isinstance(val, dict):
                val = val.get(part)
            else:
                return None
            if val is None:
                return None
        return val

    def _find_block(self, nodes, start_idx, open_tag, close_tag):
        """Find matching close tag, handling nesting."""
        depth = 1
        body = []
        i = start_idx + 1
        while i < len(nodes):
            if nodes[i][0] == open_tag:
                depth += 1
            elif nodes[i][0] == close_tag:
                depth -= 1
                if depth == 0:
                    return body, i
            body.append(nodes[i])
            i += 1
        return body, i
