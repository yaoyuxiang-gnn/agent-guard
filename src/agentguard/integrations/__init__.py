"""Optional framework integrations.

agent-guard's runtime stays dependency-free: an integration imports its
framework lazily, inside the factory function, so importing ``agentguard``
never pulls in LangChain, CrewAI or anything else. Each integration module is
self-contained and safe to import even when the framework is not installed —
the import only fails if you actually call the factory without the framework
present, and even then the fallback is a duck-typed class, not an ImportError.
"""
