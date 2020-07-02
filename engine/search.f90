subroutine find_first(needle, haystack, haystack_length, index)
    implicit none
    integer, intent(in) :: needle
    integer, intent(in) :: haystack_length
    integer, intent(in), dimension(haystack_length) :: haystack
!f2py intent(inplace) haystack
    integer, intent(out) :: index
    integer :: k
    index = -1
    do k = 1, haystack_length
        if (haystack(k)==needle) then
            index = k - 1
            exit
        endif
    enddo
end
