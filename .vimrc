" viviaxenov

" Vundle&Plugins {{{
set nocompatible              " be iMproved, required
filetype off                  " required

" set the runtime path to include Vundle and initialize
set rtp+=~/.vim/bundle/Vundle.vim
call vundle#begin()
" alternatively, pass a path where Vundle should install plugins
"call vundle#begin('~/some/path/here')

" let Vundle manage Vundle, required
Plugin 'VundleVim/Vundle.vim'

" The following are examples of different formats supported.
" Keep Plugin commands between vundle#begin/end.
" plugin on GitHub repo
Plugin 'tpope/vim-fugitive'
" Filesystem explorer
Plugin 'scrooloose/nerdtree'
" make NERDTree work with tabs
Bundle 'jistr/vim-nerdtree-tabs'
" Surround with braces etc
Plugin 'tpope/vim-surround'
" Status bar in the bottom
Plugin 'bling/vim-airline'
" Python code completion
Plugin 'davidhalter/jedi-vim'
" fuzzy completion
" The bang version will try to download the prebuilt binary if cargo does not exist.
" Plugin 'liuchengxu/vim-clap', { 'do': ':Clap install-binary!' }
" Seamless switching between vim and tmux
Plugin 'christoomey/vim-tmux-navigator'

" Colorschemes and themes {{{
Plugin 'vim-airline/vim-airline-themes'
Plugin 'nightsense/carbonized'
Plugin 'gilgigilgil/anderson.vim'
Plugin 'wadackel/vim-dogrun'
Plugin 'ajmwagar/vim-deus'
"}}}
" All of your Plugins must be added before the following line
call vundle#end()            " required
filetype plugin indent on    " required
" To ignore plugin indent changes, instead use:
"filetype plugin on
"
" Brief help
" :PluginList       - lists configured plugins
" :PluginInstall    - installs plugins; append `!` to update or just :PluginUpdate
" :PluginSearch foo - searches for foo; append `!` to refresh local cache
" :PluginClean      - confirms removal of unused plugins; append `!` to auto-approve removal
"
" see :h vundle for more details or wiki for FAQ
" Put your non-Plugin stuff after this line
" }}}

" NERDTree config {{{
let g:nerdtree_tabs_open_on_console_startup=1
let g:nerdtree_tabs_focus_on_files=1
" }}}

" Airline config {{{
"let g:airline_powerline_fonts = 1 "Включить поддержку Powerline шрифтов
"let g:airline#extensions#keymap#enabled = 0 "Не показывать текущий маппинг
"let g:airline_section_z = "\ue0a1:%l/%L Col:%c" "Кастомная графа положения курсора
"let g:Powerline_symbols='unicode' "Поддержка unicode
"let g:airline#extensions#xkblayout#enabled = 0 "Про это позже расскажу
" }}}

" Colors {{{
syntax enable           " enable syntax processing
"colorscheme peachpuff
colors deus
"set termguicolors
" }}}

" Spaces & Tabs {{{
set tabstop=4           " 4 space tab
set expandtab           " use spaces for tabs
set softtabstop=4       " 4 space tab
set shiftwidth=4
set modeline
set modelines=1
filetype indent on
filetype plugin on
set autoindent
set wildmenu
filetype indent on
" }}}

" UI Layout {{{
set number              " show line numbers
:hi CursorLine cterm=NONE ctermbg=darkgrey guibg=darkred guifg=white
set cursorline
set wildmenu
set lazyredraw
set showmatch           " higlight matching parenthesis [](){} etc.
set fillchars+=vert:┃
" }}}

" Searching {{{
set incsearch           " search as characters are entered
set hlsearch            " highlight all matches
" turn off search highlight
nnoremap <CR> :noh<CR>
" }}}

" Folding {{{
"=== folding ===
set foldmethod=indent   " fold based on indent level
set foldnestmax=10      " max 10 depth
set foldenable          " don't fold files by default on open
nnoremap <space> za
set foldlevelstart=10   " start with fold level of 1
" }}}
"

" Custom key bindings {{{
nnoremap <F9> :! python -m black % <CR>

"}}}

" vim: foldmethod=marker:foldlevel=0
