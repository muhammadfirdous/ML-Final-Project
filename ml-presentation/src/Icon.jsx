import { DynamicIcon } from 'lucide-react/dynamic';

function toKebab(s) {
  if (!s) return '';
  return s
    .replace(/([a-z])([A-Z])/g, '$1-$2')
    .replace(/([A-Z]+)([A-Z][a-z])/g, '$1-$2')
    .replace(/([a-zA-Z])(\d)/g, '$1-$2')
    .toLowerCase();
}

export function Icon({ name, size = 18, className = '', strokeWidth = 1.75, style }) {
  const kebab = toKebab(name);
  return (
    <span
      className={`inline-flex items-center justify-center ${className}`}
      style={{ width: size, height: size, ...style }}
    >
      <DynamicIcon name={kebab} size={size} strokeWidth={strokeWidth} />
    </span>
  );
}
