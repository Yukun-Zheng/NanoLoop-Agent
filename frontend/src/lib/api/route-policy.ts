type RouteRule = {
  pattern: RegExp;
  methods: ReadonlySet<string>;
};

const rules: RouteRule[] = [
  { pattern: /^health$/, methods: new Set(["GET"]) },
  { pattern: /^analyses$/, methods: new Set(["POST"]) },
  { pattern: /^analyses\/[^/]+$/, methods: new Set(["GET"]) },
  { pattern: /^analyses\/[^/]+\/export$/, methods: new Set(["GET"]) },
  {
    pattern: /^analyses\/[^/]+\/images\/[^/]+\/boxes$/,
    methods: new Set(["GET", "PUT"])
  },
  { pattern: /^analyses\/[^/]+\/queries$/, methods: new Set(["GET"]) },
  { pattern: /^analyses\/[^/]+\/query$/, methods: new Set(["POST"]) },
  { pattern: /^analyses\/[^/]+\/runs$/, methods: new Set(["POST"]) },
  { pattern: /^models$/, methods: new Set(["GET"]) },
  { pattern: /^models\/recommend$/, methods: new Set(["POST"]) },
  { pattern: /^runs\/[^/]+$/, methods: new Set(["GET"]) },
  { pattern: /^runs\/[^/]+\/corrected-mask$/, methods: new Set(["POST"]) },
  { pattern: /^runs\/[^/]+\/review$/, methods: new Set(["POST"]) },
  { pattern: /^knowledge\/documents$/, methods: new Set(["GET", "POST"]) },
  { pattern: /^knowledge\/documents\/[^/]+$/, methods: new Set(["PATCH"]) },
  { pattern: /^knowledge\/reindex$/, methods: new Set(["POST"]) },
  { pattern: /^files\/[^/]+$/, methods: new Set(["GET"]) }
];

function isSafePath(path: string): boolean {
  return (
    Boolean(path) &&
    path.length <= 8192 &&
    !path.includes("..") &&
    !path.includes("\\") &&
    !path.includes("%") &&
    !path.startsWith("/") &&
    !/[\u0000-\u001f\u007f?#]/.test(path) &&
    path.split("/").every((segment) => segment.length > 0 && segment.length <= 4096)
  );
}

export function isKnownProxyPath(path: string): boolean {
  return isSafePath(path) && rules.some((rule) => rule.pattern.test(path));
}

export function isAllowedProxyRequest(path: string, method: string): boolean {
  return (
    isKnownProxyPath(path) &&
    rules.some((rule) => rule.pattern.test(path) && rule.methods.has(method.toUpperCase()))
  );
}
