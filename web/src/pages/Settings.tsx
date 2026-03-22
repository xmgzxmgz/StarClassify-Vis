import { Moon, Sun } from 'lucide-react';
import { useTheme } from '../context/ThemeContext';

export function SettingsPage() {
  const { theme, toggleTheme } = useTheme();

  return (
    <div className="min-h-screen bg-white dark:bg-slate-950 transition-colors">
      <div className="max-w-2xl mx-auto px-4 py-8">
        {/* 页面标题 */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-slate-900 dark:text-white mb-2">
            设置
          </h1>
          <p className="text-slate-600 dark:text-slate-400">
            管理应用程序的偏好设置
          </p>
        </div>

        {/* 高级功能部分 */}
        <div className="space-y-6">
          <div className="bg-slate-50 dark:bg-slate-900 rounded-lg border border-slate-200 dark:border-slate-800 p-6">
            <h2 className="text-xl font-semibold text-slate-900 dark:text-white mb-6">
              高级功能
            </h2>

            {/* 主题切换 */}
            <div className="space-y-4">
              <div className="flex items-center justify-between p-4 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700 hover:border-slate-300 dark:hover:border-slate-600 transition-colors">
                <div className="flex items-center gap-3">
                  {theme === 'light' ? (
                    <Sun className="w-5 h-5 text-amber-500" />
                  ) : (
                    <Moon className="w-5 h-5 text-indigo-400" />
                  )}
                  <div>
                    <p className="font-medium text-slate-900 dark:text-white">
                      外观主题
                    </p>
                    <p className="text-sm text-slate-600 dark:text-slate-400">
                      {theme === 'light' ? '浅色模式' : '深色模式'}
                    </p>
                  </div>
                </div>

                {/* 切换按钮 */}
                <button
                  onClick={toggleTheme}
                  className="relative inline-flex h-8 w-14 items-center rounded-full bg-slate-300 dark:bg-slate-600 transition-colors focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 dark:focus:ring-offset-slate-950"
                  aria-label="切换主题"
                >
                  <span
                    className={`inline-block h-6 w-6 transform rounded-full bg-white shadow-lg transition-transform ${
                      theme === 'dark' ? 'translate-x-7' : 'translate-x-1'
                    }`}
                  />
                </button>
              </div>

              {/* 主题说明 */}
              <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
                <p className="text-sm text-blue-900 dark:text-blue-200">
                  💡 {theme === 'light' 
                    ? '浅色模式适合在明亮环境下使用，减少眼睛疲劳。' 
                    : '深色模式适合在暗光环境下使用，保护眼睛。'}
                </p>
              </div>
            </div>
          </div>

          {/* 其他设置项占位符 */}
          <div className="bg-slate-50 dark:bg-slate-900 rounded-lg border border-slate-200 dark:border-slate-800 p-6">
            <h2 className="text-xl font-semibold text-slate-900 dark:text-white mb-4">
              其他设置
            </h2>
            <p className="text-slate-600 dark:text-slate-400">
              更多设置选项即将推出...
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
